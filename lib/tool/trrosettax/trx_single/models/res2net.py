import os
import sys
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from .modules import Res2Net, Symm, empty_cache
from einops.layers.torch import Rearrange
from collections import namedtuple

from trx_single.utils.utils_data import esmmsa_to_mymsa


class DistPredictorSeq(nn.Module):
    def __init__(self, n_feats=16 ** 2 + 660):
        super(DistPredictorSeq, self).__init__()
        self.linear1 = nn.Linear(1280 + 20, 128)
        self.linear2 = nn.Linear(128, 16)
        self.res2net = Res2Net(in_channel=n_feats, layers=[15, 15], expansion=2)
        self.out_layer = nn.ModuleDict(
            {
                'dist':
                    nn.Sequential(*[
                        Symm('b m n d-> b n m d'),
                        nn.Linear(256, 37)
                    ])
                ,
                'omega':
                    nn.Sequential(*[
                        Symm('b m n d-> b n m d'),
                        nn.Linear(256, 25)
                    ])
                ,
                'theta': nn.Linear(256, 25),
                'phi': nn.Linear(256, 13),
            })

    def get_f2d(self, seq, emb_out, mask=None):
        device = seq.device
        if mask is not None:
            L = seq.size(-1)
            all_ind = torch.ones_like(seq)
            shuffled_ind = torch.randperm(L)
            mask_ind = shuffled_ind[:int(mask * L)]
            all_ind[:, mask_ind] = 0
            seq = torch.where(all_ind == 1, seq, torch.tensor(20, device=device))

        emb_repr = emb_out['representations'][None]  # 1, L,1280
        attn = emb_out['attentions'][None]
        # attn = Rearrange('b l h m n -> b m n (l h)')(attn)[:, 1:-1, 1:-1, :]  # 1, L,L,660
        if mask is not None:
            aa_pred = emb_out['logits'][None]  # 1,L,33
            pred_seq = torch.clone(aa_pred)
            aa_pred[..., :4] = -torch.inf
            aa_pred[..., -2:] = -torch.inf
            seq = torch.where(all_ind == 1, seq, esmmsa_to_mymsa(aa_pred.argmax(-1)))

        seq1hot = (torch.arange(20, device=device) == seq[..., None]).float()

        # 1D features
        f1d = torch.cat([seq1hot, emb_repr], dim=-1)

        if mask is not None:
            return attn, f1d, pred_seq
        return attn, f1d

    def forward(self, seq, emb_out, mask=None, is_training=False):
        fout = self.get_f2d(seq, emb_out, mask=mask)
        if mask is None:
            f2d, f1d = fout
        else:
            f2d, f1d, pred_seq = fout
        f1d = self.linear2(self.linear1(f1d))
        empty_cache()
        f1d_2d = torch.einsum('b i d, b j c -> b i j d c', f1d, f1d)
        f1d_2d = Rearrange('b i j d c -> b i j (d c)')(f1d_2d)
        # f1d_2d = torch.cat([f2d, f1d_2d], dim=-1)
        f1d_2d = Rearrange('b m n d -> b d m n')(f1d_2d)
        f2d = Rearrange('b m n d -> b d m n')(f2d)
        logits = self.res2net(f2d, f1d_2d, is_training=is_training)
        logits = Rearrange('b d m n -> b m n d')(logits)
        pred_distograms = dict(
            (k, layer(logits).softmax(-1)[0])
            for (k, layer) in self.out_layer.items()
        )
        if mask is not None:
            return pred_distograms, pred_seq
        return pred_distograms


class DistPredictorSeqRecycle(DistPredictorSeq):
    def __init__(self, n_feats=64 ** 2 + 660, esm=False):
        super(DistPredictorSeqRecycle, self).__init__(n_feats=n_feats)
        self.bhwc2bchw = Rearrange('b h w c->b c h w')
        self.bchw2bhwc = Rearrange('b c h w->b h w c')
        self.cycle_embedding = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64)
        )

    def forward(self, seq, emb_out):
        n_cycle = 4
        prev = None
        for c in range(n_cycle):
            if c == 0:
                f2d, f1d = self.get_f2d(seq, emb_out)
                f1d_ = self.linear1(f1d)
                f1d_2d = torch.einsum('b i d, b j c -> b i j d c', f1d_, f1d_)
                f1d_2d = Rearrange('b i j d c -> b i j (d c)')(f1d_2d)
                f2d_ = torch.cat([f2d, f1d_2d], dim=-1)
                f2d_ = self.bhwc2bchw(f2d_)
            else:
                prev = self.cycle_embedding(logits)
                prev = self.bhwc2bchw(prev)

            logits = self.res2net(f2d_, residue=prev)
            logits = self.bchw2bhwc(logits)
            if c != n_cycle - 1:
                logits = logits.detach()

        pred_distograms = dict(
            (k, layer(logits).softmax(-1)[0])
            for (k, layer) in self.out_layer.items()
        )
        return pred_distograms

        # pred_distograms = namedtuple('pred_distograms', ['dist', 'omega', 'phi', 'theta'])
        # results = pred_distograms(
        #     dist=self.out_layer['dist'](logits).softmax(-1)[0],
        #     omega=self.out_layer['omega'](logits).softmax(-1)[0],
        #     phi=self.out_layer['phi'](logits).softmax(-1)[0],
        #     theta=self.out_layer['theta'](logits).softmax(-1)[0],
        # )
        # return results
