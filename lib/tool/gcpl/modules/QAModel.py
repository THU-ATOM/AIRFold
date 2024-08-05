import torch
from torch.nn import functional as F
from lib.tool.gcpl.modules.QANet import *
import torch.nn as nn
from lib.tool.gcpl.modules.TriangleGraphTransformer import *
from lib.tool.gcpl.modules.IPATransformer import IPAEncoder,IPATransformer
from lib.tool.gcpl.modules.QA_utils.coordinates import get_ideal_coords
from lib.tool.gcpl.modules.QA_utils.utils import *
from lib.tool.gcpl.modules.utils import *
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from lib.tool.gcpl.modules.QA_utils.general import exists
from lib.tool.gcpl.modules.QA_utils.interface import ModelOutput

from lib.utils.systool import get_available_gpus


# Network architecture
class QA(torch.nn.Module):

    def __init__(self,

                 bert_feat_dim = 1280,
                 node_dim=64,
                 gt_heads = 32,
                 bert_attn_dim= 144,
                 protein_size = None,
                 num_channel  = 128,
                 num_restype  = 20,
                 name = None,
                 block_cycle =3,
                 verbose=False):
        super().__init__()

        self.protein_size = protein_size
        self.num_channel  = num_channel
        self.num_restype  = num_restype
        self.name = name
        self.verbose = verbose
        self.block_cycle = block_cycle

        self.bert_feat_dim = bert_feat_dim
        self.node_dim = node_dim
        self.gt_heads = gt_heads
        self.bert_attn_dim = bert_attn_dim
        self.gt_heads = 4
        self.gt_depth = 4
        # str ipa
        self.str_ipa_depth = 4
        self.str_ipa_heads = 8
        # err ipa
        self.dev_ipa_depth = 2
        self.dev_ipa_heads = 4

        #

        device_ids = get_available_gpus(1)
        self.device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'
        # self.device = "cpu"
        # protein Three dimensional information Emb

        self.model_feature=Protein_feature(num_embeddings=16)

        # 3D Convolutions.
        self.voxel_emb = Voxel(self.num_restype, 20)


        # Embedding
        self.feat_emb = nn.Sequential(
            nn.Linear(1235, self.num_channel // 2),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim)
        )

        # model_feature
        self.feat_model = nn.Sequential(
            nn.Linear(1363, self.num_channel),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim*2)
        )

        self.err_ipa = IPAEncoder(
            dim=self.node_dim*2,
            depth=self.dev_ipa_depth,
            heads=self.dev_ipa_heads,
            require_pairwise_repr=False,
        )

        self.conv2d_error = torch.nn.Conv2d(self.num_channel, 15, 1, padding=0, bias=True)
        self.conv2d_mask = torch.nn.Conv2d(self.num_channel, 1, 1, padding=0, bias=True)

        self._2d_emb=nn.Sequential(
            nn.Linear(34, self.num_channel//2),
            nn.ReLU(),
            nn.LayerNorm(self.num_channel//2)
        )

        self.qa_model = nn.Sequential(
            Transformer_ResNet(
                num_channel, 3, "model_resnet",
                inorm=True, initial_projection=True,
                extra_blocks=False
            ),
            nn.GELU(),
        )

        self.err_model = nn.Sequential(
            Transformer_ResNet(
                num_channel, 1, "model_resnet",
                inorm=True, initial_projection=False,
                extra_blocks=False
            ),
            nn.GELU(),
        )

        self.mask_model = nn.Sequential(
            Transformer_ResNet(
                num_channel, 1, "model_resnet",
                inorm=True, initial_projection=False,
                extra_blocks=False
            ),
            nn.GELU(),
        )

        self.plddt_loss1 = nn.MSELoss()
        self.str_node_transform = nn.Sequential(
            nn.Linear(
                self.bert_feat_dim,
                self.node_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.merge = TriangleGraphTransformer(
            dim=self.node_dim*2,
            edge_dim=self.node_dim*2,
            depth=2,
            tri_dim_hidden=2 * self.node_dim,
            gt_depth=self.gt_depth,
            gt_heads=self.gt_heads,
            gt_dim_head=self.node_dim // 2,
        )



        self.structure_ipa = IPATransformer(
            dim=self.node_dim*2,
            depth=self.str_ipa_depth,
            heads=self.str_ipa_heads,
            require_pairwise_repr=False,
        )

        self.dev_node_transform = nn.Sequential(
            nn.Linear(self.bert_feat_dim, self.node_dim*2),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim*2),
        )

        self.dev_edge_qa = nn.Sequential(
            nn.Linear(
                34,
                self.node_dim*2,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim*2),
        )

        # egnn
        self.egnn_feature = Protein_feature()

        self.e_node_emb = nn.Sequential(
            nn.Linear(131, self.num_channel // 2),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim)
        )

        for i in range(self.block_cycle):
            self.add_module("global_att_%i"%i,GlobalAttention(dim=64))
            self.add_module("egnn_%i"%i, EGNN(64))


    def get_coords_tran_rot(
        self,
        temp_coords,
        batch_size,
        seq_len,
        center=True,
    ):
        res_coords = rearrange(
            temp_coords,
            "b (l a) d -> b l a d",
            l=seq_len,
        )
        res_ideal_coords = repeat(
            get_ideal_coords(center=center),
            "a d -> b l a d",
            b=batch_size,
            l=seq_len,
        ).to(self.device)

        _, rotations, translations = kabsch(  
            res_ideal_coords,   
            res_coords,
            return_translation_rotation=True,
        )
        
        translations = rearrange(
            translations,
            "b l () d -> b l d",
        )
        return translations, rotations
  

    def forward(self, idx, val, obt, tbt, bert_feat, model_coords, coords_label=None,lddt_ture=None):

        nres               = obt.shape[0]
        # print("nres",nres)
        # msa_emb
        # batch_mask = torch.ones((1, nres*4), dtype=bool).cuda()  # torch.Size([1, L])
        batch_mask = torch.ones((1, nres*4), dtype=bool) # torch.Size([1, L])
        str_nodes = self.str_node_transform(bert_feat)  # linear embedding

        # model_emb
        voxel_emb = self.voxel_emb(idx, val, nres)
        _1d_feature = torch.cat((voxel_emb, obt), axis=1).unsqueeze(0)

        feat_emb = self.feat_emb(_1d_feature.detach()).permute(0,2,1)
        str_nodes = torch.cat([str_nodes, feat_emb.permute(0, 2, 1)], dim=-1)

        # refinement
        # print("Size of model_coords: ", model_coords.size())
        # print("Size of model_coords.squeeze(0): ", model_coords.squeeze(0).size())
        # model_translations, model_rotations = self.get_coords_tran_rot(model_coords.squeeze(0),batch_size=1,seq_len=nres)
        model_translations, model_rotations = self.get_coords_tran_rot(model_coords,batch_size=1,seq_len=nres)

        model_quaternion = matrix_to_quaternion(model_rotations)

        ipa_coords, ipa_translations, ipa_quaternions = self.structure_ipa(
            str_nodes,
            translations=model_translations,
            quaternions=model_quaternion,
            # pairwise_repr=str_edges,
        )

        ipa_rotations = quaternion_to_matrix(ipa_quaternions)

        # egnn make more fine
        pos_emb, AD_features, O_features, gs_d, D, E_idx = self.egnn_feature(ipa_coords.squeeze(0).permute(1,0,2))  # # 1,L,L,16;  1,L,3; 1,L,L,7, 1,L,L,15
        node_rep=str_nodes.detach().repeat(5, 1, 1)

        egnn_e = torch.cat([pos_emb, gs_d],dim=-1)
        egnn_f = torch.cat([AD_features, node_rep], dim=-1)  # 是否会存在问题
        ipa_coords = ipa_coords.squeeze(0).permute(1, 0, 2)
        egnn_f=self.e_node_emb(egnn_f)

        for i in range(self.block_cycle):
            egnn_f = self._modules["global_att_%i"% i](egnn_f)
            egnn_f, ipa_coords = self._modules["egnn_%i" % i](egnn_f, ipa_coords, egnn_e, E_idx)

        ipa_coords_T = ipa_coords
        ipa_coords = ipa_coords.permute(1, 0, 2).unsqueeze(0)

        # QA-model
        rep_translations, rep_rotations = self.get_coords_tran_rot(ipa_coords[:, :, :4].reshape(1, -1, 3).detach(), batch_size=1,seq_len=nres)
        dev_nodes = self.dev_node_transform(bert_feat)
        dev_edges = self.dev_edge_qa(tbt.permute(0,2,3,1))

        dev_out_feats = self.err_ipa(
            dev_nodes,
            translations=rep_translations,
            rotations=rep_rotations,
        )

        qa_f1d = torch.cat([dev_out_feats, _1d_feature],dim=-1)

        feat_model = self.feat_model(qa_f1d).permute(0, 2, 1)
        f1_model = tile(feat_model.unsqueeze(3), 3, nres)  # [1, 87, L, L]
        f2_model = tile(feat_model.unsqueeze(2), 2, nres)  # [1, 87, L, L]

        f2d_model = torch.cat([f1_model, f2_model, tbt, dev_edges.permute(0,3,1,2)], dim=1)

        qa_2d=self.qa_model(f2d_model)

        x_2d_err = self.err_model(qa_2d)
        deviation_logits = self.conv2d_error(x_2d_err)
        deviation_logits = (deviation_logits + deviation_logits.permute(0, 1, 3, 2)) / 2  # LL
        deviation_prediction = F.softmax(deviation_logits, dim=1)[0]

        out_mask_predictor = self.mask_model(qa_2d)
        mask_logits = self.conv2d_mask(out_mask_predictor)[:, 0, :, :]  # Reduce the second dimension
        mask_logits = (mask_logits + mask_logits.permute(0, 2, 1)) / 2
        mask_prediction = torch.sigmoid(mask_logits)[0]

        lddt_prediction = calculate_LDDT(deviation_prediction, mask_prediction)

        bb_coords = rearrange(
            ipa_coords[:, :, :3],
            "b l a d -> b (l a) d",
        )

        flat_coords = rearrange(
            ipa_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )

        # loss compute
        loss = None
        if exists(coords_label):
            rmsd_clamp = 0    #  真实的rmsd  ???
            coords_loss = kabsch_mse(
                flat_coords,
                coords_label,
                clamp=rmsd_clamp,
            )

            bb_coords_label = rearrange(
                rearrange(coords_label, "b (l a) d -> b l a d", a=4)[:, :, :3],
                "b l a d -> b (l a) d")
            bb_batch_mask = rearrange(
                rearrange(batch_mask, "b (l a) -> b l a", a=4)[:, :, :3],
                "b l a -> b (l a)")

            bondlen_loss = bond_length_l1(
                bb_coords,
                bb_coords_label,
                bb_batch_mask,
            )

            p_lddt_loss = self.plddt_loss1(lddt_prediction.to(torch.float32), lddt_ture.to(torch.float32))

            coords_loss, bondlen_loss = list(
                map(
                    lambda l: rearrange(l, "(c b) -> b c", b=1).mean(
                        1),
                    [coords_loss, bondlen_loss],
                ))

            loss = coords_loss + bondlen_loss + p_lddt_loss
        else:
            p_lddt_loss, coords_loss, bondlen_loss = None, None, None

        if not exists(coords_label):
            loss = None

        output = ModelOutput(
            coords=ipa_coords,
            p_lddt_pred=lddt_prediction,
            translations=ipa_translations,
            rotations=ipa_rotations,
            coords_loss=coords_loss,
            bondlen_loss=bondlen_loss,
            p_lddt_loss=p_lddt_loss,
            loss=loss,
        )

        return output, mask_logits, deviation_logits













