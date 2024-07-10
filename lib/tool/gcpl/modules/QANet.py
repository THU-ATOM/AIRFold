import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from torch import broadcast_tensors


class EGNN(nn.Module):
    def __init__(
            self,
            dim,
            dropout=0.5,
            init_eps=1e-3,
            norm_coors_scale_init=1e-2,
            m_pool_method='sum',
            coor_weights_clamp_value=None,
            flag=True
    ):
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        self.m_pool_method = m_pool_method
        self.flag = flag
        if self.flag == True:
            in_dim = 168
        else:
            in_dim = 286

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.node_norm = nn.LayerNorm(dim)
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim * 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim * 4, 1)
        )
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std=self.init_eps)

    def gather_nodes(self, nodes, neighbor_idx):

        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])

        return neighbor_features

    def gather_edges(self, edges, neighbor_idx):  # ###############################？？
        # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
        neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
        edge_features = torch.gather(edges, 2, neighbors)
        return edge_features

    def fourier_encode_euclidean(self, x, num_encodings=4, include_self=True):  # 傅里叶变换

        x = x.unsqueeze(-1)
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
        x = x / scales
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1) if include_self else x
        return x

    def forward(self, feats, space, edges, E_idx):

        feats_i = feats.unsqueeze(2)
        feats_j = self.gather_nodes(feats, E_idx)
        vector = torch.unsqueeze(space, 1) - torch.unsqueeze(space, 2)
        euclidean = (vector ** 2).sum(dim=-1, keepdim=True)

        if self.flag == False:
            edges = self.gather_edges(edges, E_idx)
        
        vector_nbr = self.gather_edges(vector, E_idx)
        euclidean_nbr = self.gather_edges(euclidean, E_idx)
        euclidean_fnbr = self.fourier_encode_euclidean(euclidean_nbr).squeeze(3)

        # message
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)
        edge_input = torch.cat((feats_i, feats_j, euclidean_fnbr, edges), dim=-1)
        m_ij = self.edge_mlp(edge_input)
        m_ij = m_ij * self.edge_gate(m_ij)

        vector_weights = self.coors_mlp(m_ij).squeeze(-1)
        vector_nbr = self.coors_norm(vector_nbr)
        space_out = torch.einsum('b i j, b i j c -> b i c', vector_weights, vector_nbr) + space

        if self.m_pool_method == 'sum':
            m_i = m_ij.sum(dim=-2)
        else:
            m_i = m_ij.mean(dim=-2)

        normed_feats = self.node_norm(feats)
        node_mlp_input = torch.cat((normed_feats, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + feats
        return node_out, space_out

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale

class GlobalAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=4,
            dim_head=32
    ):
        super().__init__()

        self.norm_seq = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        res_x = x
        x = self.norm_seq(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.ff(x) + res_x

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)

        self.bias_g = nn.Linear(dim, self.heads)
        self.gate_v = nn.Linear(dim, inner_dim)

    def forward(self, x):
        h, L, C , res_type= self.heads, x.shape[1], x.shape[2],x.shape[0]
        q = self.q(x).view(res_type, L, self.heads, 2 * C // self.heads).permute(0, 2, 1, 3)
        k = self.k(x).view(res_type, L, self.heads, 2 * C // self.heads).permute(0, 2, 1, 3)
        v = self.v(x).view(res_type, L, self.heads, 2 * C // self.heads).permute(0, 2, 1, 3)

        gate_values = torch.sigmoid(self.gate_v(x).view(res_type, L, 2 * C))
        bias = self.bias_g(x).permute(0, 2, 1)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v) + bias.unsqueeze(-1)
        out = out.permute(0, 2, 1, 3).reshape(res_type, L, 2 * C)
        out *= gate_values

        return self.to_out(out)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):

        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





class Voxel(torch.nn.Module):
    def __init__(self, num_restype, dim):
        super(Voxel, self).__init__()
        self.num_restype = num_restype
        self.dim = dim
        self.add_module("retype", torch.nn.Conv3d(self.num_restype, self.dim, 1, padding=0,
                                                  bias=False))  # in_channels, out_channels, kernel_size,
        self.add_module("conv3d_1", torch.nn.Conv3d(20, 20, 3, padding=0, bias=True))
        self.add_module("conv3d_2", torch.nn.Conv3d(20, 30, 4, padding=0, bias=True))
        self.add_module("conv3d_3", torch.nn.Conv3d(30, 10, 4, padding=0, bias=True))
        self.add_module("pool3d_1", torch.nn.AvgPool3d(kernel_size=4, stride=4, padding=0))

    def forward(self, idx, val, nres):
        print(idx.shape, val.shape, nres)
        x = scatter_nd(idx, val, (nres, 24, 24, 24, self.num_restype))
        x = x.permute(0, 4, 1, 2, 3)
        out_retype = self._modules["retype"](x)
        out_conv3d_1 = F.elu(self._modules["conv3d_1"](out_retype))
        out_conv3d_2 = F.elu(self._modules["conv3d_2"](out_conv3d_1))
        out_conv3d_3 = F.elu(self._modules["conv3d_3"](out_conv3d_2))
        out_pool3d_1 = self._modules["pool3d_1"](out_conv3d_3)
        voxel_emb = torch.flatten(out_pool3d_1.permute(0, 2, 3, 4, 1), start_dim=1, end_dim=-1)
        return voxel_emb


def scatter_nd(indices, updates, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = np.prod(shape)
    out = torch.zeros(size).to(device)

    temp = np.array([int(np.prod(shape[i + 1:])) for i in range(len(shape))])
    flattened_indices = torch.mul(indices.long(), torch.as_tensor(temp, dtype=torch.long).to(device)).sum(dim=1)
    print(flattened_indices.shape, updates.shape)
    out = out.scatter_add(0, flattened_indices, updates)
    return out.view(shape)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ResNet(torch.nn.Module):

    def __init__(self,
                 num_channel,
                 num_chunks,
                 name,
                 inorm=False,
                 initial_projection=False,
                 extra_blocks=False,
                 dilation_cycle=[1, 2, 4, 8, 16],
                 verbose=False):

        self.num_channel = num_channel
        self.num_chunks = num_chunks
        self.name = name
        self.inorm = inorm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = dilation_cycle
        self.verbose = verbose

        super(ResNet, self).__init__()

        if self.initial_projection:
            self.add_module("resnet_%s_init_proj" % (self.name), torch.nn.Conv2d(128, num_channel, 1))

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:
                    self.add_module("resnet_%s_%i_%i_inorm_1" % (self.name, i, dilation_rate),
                                    torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_%i_%i_inorm_2" % (self.name, i, dilation_rate),
                                    torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_%i_%i_inorm_3" % (self.name, i, dilation_rate),
                                    torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True))

                self.add_module("resnet_%s_%i_%i_conv2d_1" % (self.name, i, dilation_rate),
                                torch.nn.Conv2d(num_channel, num_channel // 2, 1))
                self.add_module("resnet_%s_%i_%i_conv2d_2" % (self.name, i, dilation_rate),
                                torch.nn.Conv2d(num_channel // 2, num_channel // 2, 3, dilation=dilation_rate,
                                                padding=dilation_rate))
                self.add_module("resnet_%s_%i_%i_conv2d_3" % (self.name, i, dilation_rate),
                                torch.nn.Conv2d(num_channel // 2, num_channel, 1))

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module("resnet_%s_extra%i_inorm_1" % (self.name, i),
                                    torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_extra%i_inorm_2" % (self.name, i),
                                    torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_extra%i_inorm_3" % (self.name, i),
                                    torch.nn.InstanceNorm2d(num_channel // 2, eps=1e-06, affine=True))

                self.add_module("resnet_%s_extra%i_conv2d_1" % (self.name, i),
                                torch.nn.Conv2d(num_channel, num_channel // 2, 1))
                self.add_module("resnet_%s_extra%i_conv2d_2" % (self.name, i),
                                torch.nn.Conv2d(num_channel // 2, num_channel // 2, 3, dilation=1, padding=1))
                self.add_module("resnet_%s_extra%i_conv2d_3" % (self.name, i),
                                torch.nn.Conv2d(num_channel // 2, num_channel, 1))

    def forward(self, x):

        if self.initial_projection:
            x = self._modules["resnet_%s_init_proj" % (self.name)](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block
                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_1" % (self.name, i, dilation_rate)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_1" % (self.name, i, dilation_rate)](x)

                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_2" % (self.name, i, dilation_rate)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_2" % (self.name, i, dilation_rate)](x)

                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_3" % (self.name, i, dilation_rate)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_3" % (self.name, i, dilation_rate)](x)
                x = x + _residual

        if self.extra_blocks:
            for i in range(2):
                _residual = x
                # Internal block
                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_1" % (self.name, i)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_1" % (self.name, i)](x)

                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_2" % (self.name, i)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_2" % (self.name, i)](x)

                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_3" % (self.name, i)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_3" % (self.name, i)](x)

                x = x + _residual

        return x

class Transformer_ResNet(torch.nn.Module):

    def __init__(self,
                 num_channel,
                 num_chunks,
                 name,
                 inorm=False,
                 initial_projection=False,
                 extra_blocks=False,
                 dilation_cycle=[1, 2, 4, 8, 16],
                 drop_path = 0.,
                 verbose=False):

        self.num_channel = num_channel
        self.num_chunks = num_chunks
        self.name = name
        self.drop_path = drop_path
        self.inorm = inorm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = dilation_cycle
        self.verbose = verbose

        super(Transformer_ResNet, self).__init__()

        if self.initial_projection:
            self.add_module("resnet_%s_init_proj" % (self.name), torch.nn.Conv2d(423-5, num_channel, 1))

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:

                if self.inorm:
                    self.add_module("resnet_%s_%i_%i_inorm_1" % (self.name, i, dilation_rate),
                                    torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_%i_%i_inorm_2" % (self.name, i, dilation_rate),
                                    torch.nn.InstanceNorm2d(num_channel * 2, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_%i_%i_inorm_3" % (self.name, i, dilation_rate),
                                    torch.nn.InstanceNorm2d(num_channel * 2, eps=1e-06, affine=True))

                self.add_module("resnet_%s_%i_%i_conv2d_1" % (self.name, i, dilation_rate),
                                torch.nn.Conv2d(num_channel, num_channel * 2, 1))
                self.add_module("resnet_%s_%i_%i_conv2d_2" % (self.name, i, dilation_rate),
                                torch.nn.Conv2d(num_channel * 2, num_channel * 2, 3, dilation=dilation_rate,
                                                padding=dilation_rate))
                self.add_module("resnet_%s_%i_%i_conv2d_3" % (self.name, i, dilation_rate),
                                torch.nn.Conv2d(num_channel * 2, num_channel, 1))
                self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

                # self.add_module("Block_%s_%i" % (self.name, i),
                #                 Block(num_channel, dilation_rate))

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module("resnet_%s_extra%i_inorm_1" % (self.name, i),
                                    torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_extra%i_inorm_2" % (self.name, i),
                                    torch.nn.InstanceNorm2d(num_channel * 2, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_extra%i_inorm_3" % (self.name, i),
                                    torch.nn.InstanceNorm2d(num_channel * 2, eps=1e-06, affine=True))

                self.add_module("resnet_%s_extra%i_conv2d_1" % (self.name, i),
                                torch.nn.Conv2d(num_channel, num_channel * 2, 1))
                self.add_module("resnet_%s_extra%i_conv2d_2" % (self.name, i),
                                torch.nn.Conv2d(num_channel * 2, num_channel * 2, 3, dilation=1, padding=1))
                self.add_module("resnet_%s_extra%i_conv2d_3" % (self.name, i),
                                torch.nn.Conv2d(num_channel * 2, num_channel, 1))
                self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                # self.add_module("Block_extra_%s_%i" % (self.name, i),
                #                 Block(num_channel, 1))


    def forward(self, x):

        if self.initial_projection:
            x = self._modules["resnet_%s_init_proj" % (self.name)](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block

                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_1" % (self.name, i, dilation_rate)](x)
                x = F.gelu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_1" % (self.name, i, dilation_rate)](x)

                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_2" % (self.name, i, dilation_rate)](x)
                x = F.gelu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_2" % (self.name, i, dilation_rate)](x)

                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_3" % (self.name, i, dilation_rate)](x)
                x = F.gelu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_3" % (self.name, i, dilation_rate)](x)
                x = self.drop_path1(x) + _residual


        if self.extra_blocks:
            for i in range(2):
                _residual = x

                # Internal block

                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_1" % (self.name, i)](x)
                x = F.gelu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_1" % (self.name, i)](x)

                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_2" % (self.name, i)](x)
                x = F.gelu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_2" % (self.name, i)](x)

                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_3" % (self.name, i)](x)
                x = F.gelu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_3" % (self.name, i)](x)
                x = self.drop_path2(x) + _residual

        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, dilation_rate, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dwconv_c1=torch.nn.Conv2d(dim, dim, 3, dilation=dilation_rate, padding=dilation_rate)

    def forward(self, x):
        input = x
        x=self.dwconv_c1(x)
        # x = self.dwconv(x)  # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class PositionalEncodings(torch.nn.Module):

    def __init__(self, num_embeddings, period_range=[2, 1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, E_idx):
        # i-j
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)

        # N_nodes = E_idx.size(0)
        # N_neighbors = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).cuda()
        d = (E_idx.float() - ii).unsqueeze(-1)

        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).cuda()

        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        return E

class Protein_feature(torch.nn.Module):

    def __init__(self, dmin=0, dmax=15, step=0.4, var=None, num_embeddings=16):
        super(Protein_feature, self).__init__()
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var
        self.pos = PositionalEncodings(num_embeddings)

    def gather_nodes(self, nodes, neighbor_idx):
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
        return neighbor_features

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)  # ??

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)

        O_neighbors = self.gather_nodes(O, E_idx)
        X_neighbors = self.gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        # O_features = torch.cat((dU), dim=-1)
        O_features = dU

        return AD_features, O_features

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)

        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _dist(self, X, top_k, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        # Identify k nearest neighbors (including self)
        D_neighbors, E_idx = torch.topk(D, top_k, dim=-1, largest=False)  # 与E_idx返回标号,在原来的位置上的索引,从原数组中从小到大获得
        return D, D_neighbors, E_idx

    def gather_edges(self, edges, neighbor_idx):  # ###############################？？
        # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
        neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
        edge_features = torch.gather(edges, 2, neighbors)
        return edge_features

    def gs_dist(self, D):
        # Distance radial basis function

        D_min, D_max, D_count = 0., 20., 15
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        gs_dist = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return gs_dist

    def forward(self, coords, num_k=30):
        # gs_d = self._gs_distance(D_neighbors)
        D, D_neighbors, E_idx = self._dist(coords, top_k=num_k)
        gs_d = self.gs_dist(D_neighbors)
        pos_emb = self.pos(E_idx)  # 1,L,L,16 爆红什么原因

        AD_features, O_features = self._orientations_coarse(coords, E_idx)

        return pos_emb.cuda(), AD_features.squeeze(0).type(torch.FloatTensor).cuda(), \
               O_features.cuda(), gs_d.cuda(), D_neighbors, E_idx