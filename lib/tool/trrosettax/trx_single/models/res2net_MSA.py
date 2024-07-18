import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DistPredictorMSA(nn.Module):
    def __init__(self, in_channel=526, n_blocks=[3, 4, 6, 3]):
        super(DistPredictorMSA, self).__init__()
        self.net = nn.Sequential(
            Res2Net(in_channel, n_blocks)
        )
        self.out_elu = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True),
            nn.ELU(inplace=True)
        )
        self.conv_d = nn.Conv2d(512, 37, 1)
        self.conv_p = nn.Conv2d(512, 13, 1)
        self.conv_t = nn.Conv2d(512, 25, 1)
        self.conv_o = nn.Conv2d(512, 25, 1)

    def forward(self, msa,return_logits=False):
        with torch.no_grad():
            f2d, f1d = self.get_f2d(msa[0])
            f2d = f2d.permute(2, 0, 1).unsqueeze(0)
        output_tensor = self.net(f2d)  # 1,512,L,L
        output_tensor = self.out_elu(output_tensor)
        symm = output_tensor + output_tensor.permute(0, 1, 3, 2)
        pred_distograms = {}
        pred_distograms['dist'] = F.softmax(self.conv_d(symm), dim=1).permute(0, 2, 3, 1)  # 1,L,L,37
        pred_distograms['omega'] = F.softmax(self.conv_o(symm), dim=1).permute(0, 2, 3, 1)
        pred_distograms['theta'] = F.softmax(self.conv_t(output_tensor), dim=1).permute(0, 2, 3, 1)
        pred_distograms['phi'] = F.softmax(self.conv_p(output_tensor), dim=1).permute(0, 2, 3, 1)
        if return_logits:
            return pred_distograms, output_tensor
        else:
            return pred_distograms

    def get_f2d(self, msa):
        nrow, ncol = msa.size()[-2:]
        msa1hot = (torch.arange(21, device=msa.device) == msa[..., None].long()).float()
        w = self.reweight(msa1hot, .8)

        # 1D features
        f1d_seq = msa1hot[0, :, :20]
        f1d_pssm = self.msa2pssm(msa1hot, w)

        f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)

        # 2D features
        f2d_dca = self.fast_dca(msa1hot, w) if nrow > 1 else torch.zeros([ncol, ncol, 442], device=msa.device)

        f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
                         f1d[None, :, :].repeat([ncol, 1, 1]),
                         f2d_dca], dim=-1)
        f2d = f2d.view([ncol, ncol, 442 + 2 * 42])
        return f2d, f1d

    @staticmethod
    def msa2pssm(msa1hot, w):
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(dim=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(dim=-1).float()
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        nr, nc, ns = msa1hot.size()
        device = msa1hot.device
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())
        mean = (x * weights[:, None]).sum(dim=0, keepdim=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns).to(device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)

        x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum((1, 3))) * (1 - torch.eye(nc).to(device))
        apc = x3.sum(dim=0, keepdim=True) * x3.sum(dim=1, keepdim=True) / x3.sum()
        contacts = (x3 - apc) * (1 - torch.eye(nc).to(device))

        return torch.cat([features, contacts[:, :, None]], dim=2)


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, baseWidth=26, scale=4, stype='normal'):
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

        self.relu = nn.ELU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        if self.stype == 'stage':
            residual = self.conv_st(residual)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class Res2Net(nn.Module):

    def __init__(self, in_channel, layers, baseWidth=26, scale=4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, affine=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channel, 64, 1),
        )
        self.layer1 = self._make_layer(Bottle2neck, 64, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, 128, layers[1])
        self.layer3 = self._make_layer(Bottle2neck, 128, layers[2])
        self.layer4 = self._make_layer(Bottle2neck, 128, layers[3])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        d = 1
        for i in range(1, blocks):
            d = 2 * d % 31
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale, dilation=d))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
