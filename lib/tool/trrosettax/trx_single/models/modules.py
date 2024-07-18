import math
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Symm(nn.Module):
    def __init__(self, pattern):
        super(Symm, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return (x + Rearrange(self.pattern)(x)) / 2


def my_norm(x, weight, bias):
    mx = x.mean(dim=[-1, -2], keepdim=True)
    sx = (x.var(dim=[-1, -2], keepdim=True) + 1e-5) ** .5
    x -= mx
    x /= sx
    x *= weight[None, :, None, None]
    x += bias[None, :, None, None]


def my_fc(*fs, fc):
    idx = np.cumsum([f.size(1) for f in fs])
    weights = torch.tensor_split(fc.weight, tuple(idx[:-1]), dim=1)
    assert len(weights) == len(fs)
    for i, (f, w) in enumerate(zip(*(fs, weights))):
        if i == 0:
            out = torch.einsum('bdij, cd->bcij', f, w.squeeze())
        else:
            out += torch.einsum('bdij, cd->bcij', f, w.squeeze())
        del f
        empty_cache()

    del fs
    empty_cache()
    out += fc.bias[None, :, None, None]
    return out


class Res2Net(nn.Module):
    def __init__(self, in_channel, layers, baseWidth=26, scale=4, expansion=4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.n_layers = layers
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, affine=True),
            nn.ELU(inplace=False),
            nn.Conv2d(in_channel, 64, 1),
        )
        channels = [64, 128, 128, 128]
        self.layers = nn.Sequential(
            *[self._make_layer(Bottle2neck, channels[i], layers[i], expansion=expansion)
              for i in range(len(layers))]
        )

    def _make_layer(self, block, planes, blocks, stride=1, expansion=4):
        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * expansion
        d = 1
        for i in range(1, blocks):
            d = 2 * d % 31
            layers.append(block(self.inplanes, planes, expansion=expansion, baseWidth=self.baseWidth, scale=self.scale, dilation=d))
        return nn.Sequential(*layers)

    def forward(self, f1, f2, return_mid=False, residue=None, is_training=False):
        if f1.size(-1) > 300 and not is_training:
            # inplace to save memory
            my_norm(f1, self.conv1[0].weight[:f1.size(1)], self.conv1[0].bias[:f1.size(1)])
            my_norm(f2, self.conv1[0].weight[-f2.size(1):], self.conv1[0].bias[-f2.size(1):])
            f1 = self.conv1[1](f1)
            f2 = self.conv1[1](f2)
            x = my_fc(f1, f2, fc=self.conv1[2])
            empty_cache()
        else:
            x = self.conv1(torch.cat([f1, f2], dim=1))
        if residue is not None:
            x = x + residue
        if return_mid:
            mid_tensors = []
            for i in range(len(self.n_layers)):
                x = self.layers[i](x)
                mid_tensors.append(x)
            return x, mid_tensors
        else:
            for group in self.layers:
                for block in group:
                    x = block(x, is_training=is_training)
                    if x.size(-1) > 300 and not is_training:
                        empty_cache()
            return x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, baseWidth=26, scale=4, stype='normal', expansion=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        self.expansion = expansion

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(inplanes, affine=True)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
            bns.append(nn.InstanceNorm2d(width, affine=True))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(width * scale, affine=True)

        self.conv_st = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1)

        self.relu = nn.ELU(inplace=False)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, is_training=False):
        residual = x
        empty_cache()
        if x.size(-1) > 300 and not is_training:
            my_norm(x, self.bn1.weight, self.bn1.bias)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        if x.size(-1) > 300 and not is_training:
            x = my_fc(x, fc=self.conv1)
        else:
            x = self.conv1(x)

        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            empty_cache()
            if i == 0:
                x = sp
            else:
                x = torch.cat((x, sp), 1)

        x = torch.cat((x, spx[self.nums]), 1)
        if not is_training:
            del spx, sp
        empty_cache()
        if self.stype == 'stage':
            if x.size(-1) > 300 and not is_training:
                residual = my_fc(residual, fc=self.conv_st)
            else:
                residual = self.conv_st(residual)
            empty_cache()
        if x.size(-1) > 300 and not is_training:
            my_norm(x, self.bn3.weight, self.bn3.bias)
        else:
            x = self.bn3(x)
        x = self.relu(x)
        if x.size(-1) > 300 and not is_training:
            x = my_fc(x, fc=self.conv3)
        else:
            x = self.conv3(x)

        if not is_training:
            x += residual
            del residual
        else:
            x = x + residual

        empty_cache()
        return x
