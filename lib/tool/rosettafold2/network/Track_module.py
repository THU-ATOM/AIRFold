import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.utils.checkpoint as checkpoint
from util import cross_product_matrix
from util_module import *
from Attention_module import *
from SE3_network import SE3TransformerWrapper
from kinematics import normQ, avgQ, Qs2Rs, Rs2Qs
from symmetry import get_symm_map


# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

# Module contains classes and functions to generate initial embeddings
class SeqSep(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32):
        super(SeqSep, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
    
    def forward(self, idx, oligo=1):
        B, L = idx.shape[:2]
        bins = torch.arange(self.minpos, self.maxpos, device=idx.device)
        seqsep = torch.full((oligo,L,L), 100, dtype=idx.dtype, device=idx.device)
        seqsep[0] = idx[:,None,:] - idx[:,:,None] # (B, L, L)
        #
        ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
        emb = self.emb(ib) #(B, L, L, d_model)
        return emb

# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16, d_rbf=64, 
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.emb_rbf = nn.Linear(d_rbf, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.emb_rbf = init_lecun_normal(self.emb_rbf)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.emb_rbf.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state, symmids, symmsub):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, d_rbf)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L, _ = msa.shape

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = pair + self.emb_rbf(rbf_feat)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.type_as(state)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa

class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_hidden_state=16, d_rbf=64, d_state=32, p_drop=0.15):
        super(PairStr2Pair, self).__init__()
        
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_left = nn.Linear(d_state, d_hidden_state)
        self.proj_right = nn.Linear(d_state, d_hidden_state)
        self.to_gate = nn.Linear(d_hidden_state*d_hidden_state, d_pair)

        self.emb_rbf = nn.Linear(d_rbf, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)
        
        self.tri_mul_out = TriangleMultiplication(d_pair, d_hidden=d_hidden)
        self.tri_mul_in = TriangleMultiplication(d_pair, d_hidden, outgoing=False)

        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb_rbf = init_lecun_normal(self.emb_rbf)
        nn.init.zeros_(self.emb_rbf.bias)
        
        self.proj_left = init_lecun_normal(self.proj_left)
        nn.init.zeros_(self.proj_left.bias)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_right.bias)
        
        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.ones_(self.to_gate.bias)

    # perform a striped p2p op
    def subblock(self, OP, pair, rbf_feat, crop):
        N,L = pair.shape[:2]

        nbox = (L-1)//(crop//2)+1
        idx = torch.triu_indices(nbox,nbox,1, device=pair.device)
        ncrops = idx.shape[1]

        pairnew = torch.zeros((N,L*L,pair.shape[-1]), device=pair.device, dtype=pair.dtype)
        countnew = torch.zeros((N,L*L), device=pair.device, dtype=torch.int)

        for i in range(ncrops):
            # reindex sub-blocks
            offsetC = torch.clamp( (1+idx[1,i:(i+1)])*(crop//2)-L, min=0 ) # account for going past L
            offsetN = torch.zeros_like(offsetC)
            mask = (offsetC>0)*((idx[0,i]+1)==idx[1,i])
            offsetN[mask] = offsetC[mask]
            pairIdx = torch.zeros((1,crop), dtype=torch.long, device=pair.device)
            pairIdx[:,:(crop//2)] = torch.arange(crop//2, dtype=torch.long, device=pair.device)+idx[0,i:(i+1),None]*(crop//2) - offsetN[:,None]
            pairIdx[:,(crop//2):] = torch.arange(crop//2, dtype=torch.long, device=pair.device)+idx[1,i:(i+1),None]*(crop//2) - offsetC[:,None]

            # do reindexing
            iL,iU = pairIdx[:,:,None], pairIdx[:,None,:]
            paircrop = pair[:,iL,iU,:].reshape(-1,crop,crop,pair.shape[-1])
            rbfcrop = rbf_feat[:,iL,iU,:].reshape(-1,crop,crop,rbf_feat.shape[-1])

            # row attn
            paircrop = OP(paircrop, rbfcrop).to(pair.dtype)

            # unindex
            iUL = (iL*L+iU).flatten()
            pairnew.index_add_(1,iUL, paircrop.reshape(N,iUL.shape[0],pair.shape[-1]))
            countnew.index_add_(1,iUL, torch.ones((N,iUL.shape[0]), device=pair.device, dtype=torch.int))

        return pair + (pairnew/countnew[...,None]).reshape(N,L,L,-1)

    def forward(self, pair, rbf_feat, state, crop=-1, symmids=None, symmsub=None):
        B,L = pair.shape[:2]

        rbf_feat = self.emb_rbf(rbf_feat)

        state = self.norm_state(state)
        left = self.proj_left(state)
        right = self.proj_right(state)
        gate = einsum('bli,bmj->blmij', left, right).reshape(B,L,L,-1)
        gate = torch.sigmoid(self.to_gate(gate))
        rbf_feat = gate*rbf_feat

        crop = 2*(crop//2) # make sure even

        if (crop>0 and crop<=L):
            pair = self.subblock( 
                lambda x,y:self.drop_row(self.tri_mul_out(x)), #, symmids)),
                pair, rbf_feat, crop
            )

            pair = self.subblock( 
                lambda x,y:self.drop_row(self.tri_mul_in(x)), #, symmids)),
                pair, rbf_feat, crop
            )

            pair = self.subblock( 
                lambda x,y:self.drop_row(self.row_attn(x,y)), #, symmids)),
                pair, rbf_feat, crop
            )

            pair = self.subblock( 
                lambda x,y:self.drop_col(self.col_attn(x,y)), #, symmids)),
                pair, rbf_feat, crop
            )

            # feed forward layer
            RESSTRIDE = 16384//L
            for i in range((L-1)//RESSTRIDE+1):
                r_i,r_j = i*RESSTRIDE, min((i+1)*RESSTRIDE,L)
                pair[:,r_i:r_j] = pair[:,r_i:r_j] + self.ff(pair[:,r_i:r_j])

        else:
            #_nc = lambda x:torch.sum(torch.isnan(x))
            pair = pair + self.drop_row(self.tri_mul_out(pair)) 
            pair = pair + self.drop_row(self.tri_mul_in(pair)) 
            pair = pair + self.drop_row(self.row_attn(pair, rbf_feat)) 
            pair = pair + self.drop_col(self.col_attn(pair, rbf_feat)) 
            pair = pair + self.ff(pair)

        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        self.d_hidden = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair, symmids, symmsub):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)

        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
       
        pair = pair + out
        
        return pair

class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.linear_out = nn.Linear(d_hidden, 20)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)
        
        # right before relu activation: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # right before residual connection: zero initialize
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq, state):
        '''
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: hidden embeddings corresponding to query sequence (B, L, d_msa)
            - state: state feature (output l0 feature) from previous SE3 layer (B, L, d_state)
        Outputs:
            - si: predicted torsion angles (phi, psi, omega, chi1~4 with cos/sin, Cb bend, Cb twist, CG) (B, L, 10, 2)
        '''
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        si = self.linear_out(F.relu_(si))
        return si.view(B, L, 10, 2)

def update_symm_Rs(Rs, Ts, Lasu, symmsub_in, symmsub, symmRs):
    def dist_error(R0,T0,Rs,Ts):
        B = Ts.shape[0]
        Tcom = Ts[:,:Lasu].mean(dim=1,keepdim=True)
        Tcorr = torch.einsum('ij,brj->bri', R0, Ts[:,:Lasu]-Tcom) + Tcom + 10.0*T0[None,None,:]
        Xsymm = torch.einsum('sij,brj->bsri', symmRs[symmsub], Tcorr).reshape(B,-1,3)
        Xtrue = Ts
        dsymm = torch.linalg.norm(Xsymm[:,:,None]-Xsymm[:,None,:], dim=-1)
        dtrue = torch.linalg.norm(Xtrue[:,:,None]-Xtrue[:,None,:], dim=-1)
        return torch.clamp( torch.abs(dsymm-dtrue), max=10.0).mean()

    B = Ts.shape[0]

    # symmetry correction 1: don't let COM (of entire complex) move
    Tmean = Ts[:,:Lasu].reshape(-1,3).mean(dim=0)
    Tmean = torch.einsum('sij,j->si', symmRs, Tmean).mean(dim=0)
    Ts = Ts - Tmean

    # symmetry correction 2: use minimization to minimize drms
    #with torch.enable_grad():
    #    T0 = torch.zeros(3,device=Ts.device).requires_grad_(True)
    #    R0 = torch.eye(3,device=Ts.device).requires_grad_(True)
    #    opt = torch.optim.SGD([T0,R0], lr=0.001)
    #
    #    if (dist_error(R0,T0,symmRs[symmsub],Ts)>0.5):
    #        for e in range(101):
    #            loss = dist_error(R0,T0,symmRs[symmsub],Ts)
    #            if (e%50 == 0):
    #                print (e,loss)
    #            opt.zero_grad()
    #            loss.backward()
    #            opt.step()
    #
    #Tcom = Ts[:,:Lasu].mean(dim=1,keepdim=True)
    #Ts = torch.einsum('ij,brj->bri', R0, Ts[:,:Lasu]-Tcom) +Tcom + 10.0*T0[None,None,:]

    Rs = torch.einsum('sij,brjk,slk->bsril', symmRs[symmsub], Rs[:,:Lasu], symmRs[symmsub_in])
    Ts = torch.einsum('sij,brj->bsri', symmRs[symmsub], Ts[:,:Lasu])
    Rs = Rs.reshape(B,-1,3,3) # (B,S,L,3,3)
    Ts = Ts.reshape(B,-1,3) # (B,S,L,3,3)
    return Rs, Ts

def update_symm_subs(Rs, Ts, pair, symmids, symmsub_in, symmsub, symmRs, metasymm):
    B,Ls = Ts.shape[0:2]
    Osub = symmsub.shape[0]
    L = Ls//Osub

    com = Ts[:,:L].sum(dim=-2)
    rcoms = torch.einsum('sij,bj->si', symmRs, com)
    subsymms, nneighs = metasymm
    symmsub_new = []
    for i in range(len(subsymms)):
        drcoms = torch.linalg.norm(rcoms[0,:] - rcoms[subsymms[i],:], dim=-1)
        _,subs_i = torch.topk(drcoms,nneighs[i],largest=False)
        subs_i,_ = torch.sort( subsymms[i][subs_i] )
        symmsub_new.append(subs_i)

    symmsub_new = torch.cat(symmsub_new)

    s_old = symmids[symmsub[:,None],symmsub[None,:]]
    s_new = symmids[symmsub_new[:,None],symmsub_new[None,:]]

    # remap old->new
    # a) find highest-magnitude patches
    pairsub = dict()
    pairmag = dict()
    for i in range(Osub):
        for j in range(Osub):
            idx_old = s_old[i,j].item()
            sub_ij = pair[:,i*L:(i+1)*L,j*L:(j+1)*L,:].clone()
            mag_ij = torch.max(sub_ij.flatten()) #torch.norm(sub_ij.flatten())
            if idx_old not in pairsub or mag_ij > pairmag[idx_old]:
                pairmag[idx_old] = mag_ij
                pairsub[idx_old] = (i,j) #sub_ij

    # b) reindex
    idx = torch.zeros((Osub*L,Osub*L),dtype=torch.long,device=pair.device)
    idx = (
        torch.arange(Osub*L,device=pair.device)[:,None]*Osub*L
         + torch.arange(Osub*L,device=pair.device)[None,:]
    )
    for i in range(Osub):
        for j in range(Osub):
            idx_new = s_new[i,j].item()
            if idx_new in pairsub:
                inew,jnew = pairsub[idx_new]
                idx[i*L:(i+1)*L,j*L:(j+1)*L] = (
                    Osub*L*torch.arange(inew*L,(inew+1)*L)[:,None]
                    + torch.arange(jnew*L,(jnew+1)*L)[None,:]
                )
    pair = pair.view(1,-1,pair.shape[-1])[:,idx.flatten(),:].view(1,Osub*L,Osub*L,pair.shape[-1])

    #for i in range(Osub):
    #    for j in range(Osub):
    #        idx_new = s_new[i,j].item()
    #        if idx_new in pairsub:
    #            pass
    #            #pair[:,i*L:(i+1)*L,j*L:(j+1)*L,:] = pairsub[idx_new] #/pairmag[idx_new]

    #if (torch.any(symmsub!=symmsub_new)):
    #    print (symmsub,'->',symmsub_new)

    if symmsub_in is not None and symmsub_in.shape[0]>1:
        Rs, Ts = update_symm_Rs(Rs, Ts, L, symmsub_in, symmsub_new, symmRs)

    return Rs, Ts, pair, symmsub_new

class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, d_rbf=64, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        
        self.embed_node1 = nn.Linear(d_msa, SE3_param['l0_in_features'])
        self.embed_node2 = nn.Linear(d_state, SE3_param['l0_in_features'])
        self.ff_node = FeedForwardLayer(SE3_param['l0_in_features'], 2, p_drop=p_drop)
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])

        self.embed_edge1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_edge2 = nn.Linear(d_rbf+1, SE3_param['num_edge_features'])
        self.ff_edge = FeedForwardLayer(SE3_param['num_edge_features'], 2, p_drop=p_drop)
        self.norm_edge = nn.LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                                   p_drop=p_drop)
        
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_node1 = init_lecun_normal(self.embed_node1)
        self.embed_node2 = init_lecun_normal(self.embed_node2)
        self.embed_edge1 = init_lecun_normal(self.embed_edge1)
        self.embed_edge2 = init_lecun_normal(self.embed_edge2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_node1.bias)
        nn.init.zeros_(self.embed_node2.bias)
        nn.init.zeros_(self.embed_edge1.bias)
        nn.init.zeros_(self.embed_edge2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair_in, R_in, T_in, xyz, state, idx_in, symmids, symmsub, symmRs, symmmeta, top_k=64, eps=1e-5):
        B, N, L = msa.shape[:3]

        # process msa & state features
        seq = self.norm_msa(msa[:,0])
        state = self.norm_state(state)

        # get node features
        node = self.embed_node1(seq) + self.embed_node2(state)
        node = node + self.ff_node(node)
        node = self.norm_node(node)
        node = node.reshape(B*L, -1, 1)

        # pair
        pair_in = self.norm_pair(pair_in)

        # get edge features
        neighbor = get_seqsep(idx_in)
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1])).reshape(B,L,L,-1)
        rbf_feat = torch.cat((rbf_feat, neighbor), dim=-1)
        edge = self.embed_edge1(pair_in) + self.embed_edge2(rbf_feat)
        edge = edge + self.ff_edge(edge)
        edge = self.norm_edge(edge)

        # define graph
        G, edge_feats = make_topk_graph(xyz[:,:,1,:].detach(), edge, idx_in, top_k=top_k)

        # extra L1 features (CA-N and CA-C vectors)
        l1_feats = torch.stack((xyz[:,:,0,:], xyz[:,:,2,:]), dim=-2)
        l1_feats = l1_feats - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)

        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, node, l1_feats, edge_feats)
        state = state + shift['0'].reshape(B, L, -1) # (B, L, C)

        offset = shift['1'].reshape(B, L, 2, 3)
        Ts = offset[:,:,0,:] * 10.0 # translation
        Qs = offset[:,:,1,:] # rotation

        Qs = torch.cat((torch.ones((B,L,1),device=Qs.device),Qs),dim=-1)
        Qs = normQ(Qs)
        Rs = Qs2Rs(Qs)

        seqfull = msa[:,0]
        alpha = self.sc_predictor(seqfull, state)

        Rs = einsum('bnij,bnjk->bnik', Rs, R_in)
        Ts = Ts + T_in 

        return Rs, Ts, state, alpha

class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_rbf=64,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.pos = SeqSep(d_rbf)
        self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                 d_hidden=d_hidden//2, p_drop=p_drop)
        self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair, d_state=SE3_param['l0_out_features'],
                                      d_hidden=d_hidden, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                               d_state=SE3_param['l0_out_features'],
                               SE3_param=SE3_param,
                               p_drop=p_drop)

    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, symmids, symmsub_in, symmsub, symmRs, symmmeta, use_checkpoint=False, topk=0, crop=-1):
        #rbf_feat = rbf(torch.cdist(xyz[:,:,1,:], xyz[:,:,1,:])) + self.pos(idx)
        O,L = pair.shape[:2]
        xyzfull = xyz.view(1,O*L,3,3)
        rbf_feat = rbf(
            torch.cdist(xyzfull[:,:,1,:], xyzfull[:,:L,1,:])
        ).reshape(O,L,L,-1) + self.pos(idx, O)

        if use_checkpoint:
            msa = checkpoint.checkpoint(create_custom_forward(self.msa2msa), msa, pair, rbf_feat, state, symmids, symmsub)
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair, symmids, symmsub)
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair, rbf_feat, state, crop, symmids, symmsub)
            R, T, state, alpha = checkpoint.checkpoint(create_custom_forward(self.str2str, top_k=topk, symmmeta=symmmeta), 
                msa.float(), pair.float(), R_in.float(), T_in.float(), xyz.float(), state.float(), idx, symmids, symmsub, symmRs)
        else:
            msa = self.msa2msa(msa, pair, rbf_feat, state, symmids, symmsub)
            pair = self.msa2pair(msa, pair, symmids, symmsub)
            pair = self.pair2pair(pair, rbf_feat, state, crop, symmids, symmsub)
            R, T, state, alpha = self.str2str(
                msa.float(), pair.float(), R_in.float(), T_in.float(), xyz.float(), state.float(), idx, symmids, symmsub, symmRs, symmmeta, top_k=topk) 

        # update contacting subunits
        # symmetrize pair features
        if symmsub is not None and symmsub.shape[0]>1:
            #print ('x1', R.shape, T.shape, pair.shape)
            R, T, pair, symmsub = update_symm_subs(R, T, pair, symmids, symmsub_in, symmsub, symmRs, symmmeta)
            #print ('x2', R.shape, T.shape, pair.shape)

        return msa, pair, R, T, state, alpha, symmsub

class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 p_drop=0.15):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block
        
        self.proj_state = nn.Linear(SE3_param_topk['l0_out_features'], SE3_param_full['l0_out_features'])
        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        p_drop=p_drop,
                                                        use_global_attn=True,
                                                        SE3_param=SE3_param_full)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       p_drop=p_drop,
                                                       use_global_attn=False,
                                                       SE3_param=SE3_param_full)
                                                       for i in range(n_main_block)])

        self.proj_state2 = nn.Linear(SE3_param_full['l0_out_features'], SE3_param_topk['l0_out_features'])
        # Final SE(3) refinement
        if n_ref_block > 0:
            self.str_refiner = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                       d_state=SE3_param_topk['l0_out_features'],
                                       SE3_param=SE3_param_topk,
                                       p_drop=p_drop)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_state.bias)
        self.proj_state2 = init_lecun_normal(self.proj_state2)
        nn.init.zeros_(self.proj_state2.bias)

    def forward(self, seq, msa, msa_full, pair, xyz_in, state, idx, symmids, symmsub, symmRs, symmmeta, use_checkpoint=False, p2p_crop=-1, topk_crop=-1):
        # input:
        #   seq: query sequence (B, L)
        #   msa: seed MSA embeddings (B, N, L, d_msa)
        #   msa_full: extra MSA embeddings (B, N, L, d_msa_full)
        #   pair: initial residue pair embeddings (B, L, L, d_pair)
        #   xyz_in: initial BB coordinates (B, L, N/CA/C, 3)
        #   state: initial state features containing mixture of query seq, sidechain, accuracy info (B, L, d_state)
        #   idx: residue index

        B,_,L = msa.shape[:3]
        if symmsub is not None:
            Lasu = L//symmsub.shape[0]
            symmsub_in = symmsub.clone()
        else:
            Lasu = L
            symmsub_in = None

        R_in = torch.eye(3, device=xyz_in.device).reshape(1,1,3,3).expand(B, L, -1, -1)
        T_in = xyz_in[:,:,1].clone()
        xyz_in = xyz_in - T_in.unsqueeze(-2)
    
        state = self.proj_state(state)

        R_s = list()
        T_s = list()
        alpha_s = list()
        for i_m in range(self.n_extra_block):
            R_in = R_in.detach() # detach rotation (for stability)
            T_in = T_in.detach() # detach rotation (for stability)
            # Get current BB structure
            xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)

            msa_full, pair, R_in, T_in, state, alpha, symmsub = self.extra_block[i_m](msa_full, pair,
                                                                             R_in, T_in, xyz, state, idx, symmids, symmsub_in, symmsub, symmRs, symmmeta,
                                                                             use_checkpoint=use_checkpoint, crop=p2p_crop, topk=topk_crop)

            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        for i_m in range(self.n_main_block):
            R_in = R_in.detach()
            T_in = T_in.detach() # detach rotation (for stability)
            # Get current BB structure
            xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
            
            msa, pair, R_in, T_in, state, alpha, symmsub = self.main_block[i_m](msa, pair,
                                                                R_in, T_in, xyz, state, idx, symmids, symmsub_in, symmsub, symmRs, symmmeta,
                                                                use_checkpoint=use_checkpoint, crop=p2p_crop, topk=topk_crop)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)
       
        state = self.proj_state2(state)
        for i_m in range(self.n_ref_block):
            R_in = R_in.detach()
            T_in = T_in.detach() # detach rotation (for stability)
            xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
            R_in, T_in, state, alpha = self.str_refiner(
                msa.float(), pair.float(), R_in.float(), T_in.float(), xyz.float(), state.float(), idx, symmids, symmsub, symmRs, symmmeta, top_k=64)

            
            if symmsub_in is not None and symmsub_in.shape[0]>1:
                R_in, T_in = update_symm_Rs(R_in, T_in, Lasu, symmsub_in, symmsub, symmRs)

            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)


        R_s = torch.stack(R_s, dim=0)
        T_s = torch.stack(T_s, dim=0)
        alpha_s = torch.stack(alpha_s, dim=0)

        return msa, pair, R_s, T_s, alpha_s, state, symmsub
