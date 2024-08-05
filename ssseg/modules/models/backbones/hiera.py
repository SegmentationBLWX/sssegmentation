'''
Function:
    Implementation of Hiera
Author:
    Zhenchao Jin
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ...utils import BaseModuleBuilder
from .bricks import BuildActivation, BuildNormalization


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''PositionEmbeddingSine'''
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000, normalize=True, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}
    '''encodexy'''
    def encodexy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y
    '''encodeboxes'''
    @torch.no_grad()
    def encodeboxes(self, x, y, w, h):
        pos_x, pos_y = self.encodexy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos
    '''Backwards compatibility'''
    encode = encodeboxes
    '''encodepoints'''
    @torch.no_grad()
    def encodepoints(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self.encodexy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos
    '''forward'''
    @torch.no_grad()
    def forward(self, x):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device).view(1, -1, 1).repeat(x.shape[0], 1, x.shape[-1])
        x_embed = torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device).view(1, 1, -1).repeat(x.shape[0], x.shape[-2], 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


'''PositionEmbeddingRandom'''
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=64, scale=None):
        super(PositionEmbeddingRandom, self).__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))
    '''peencoding'''
    def peencoding(self, coords):
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    '''forward'''
    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self.peencoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)
    '''forwardwithcoords'''
    def forwardwithcoords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self.peencoding(coords.to(torch.float))


'''PEBuilder'''
class PEBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'PositionEmbeddingSine': PositionEmbeddingSine, 'PositionEmbeddingRandom': PositionEmbeddingRandom
    }
    '''build'''
    def build(self, pe_cfg):
        return super().build(pe_cfg)


'''BuildPE'''
BuildPE = PEBuilder().build


'''windowpartition'''
def windowpartition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


'''windowunpartition'''
def windowunpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


'''dopool'''
def dopool(x, pool, norm=None):
    # None, direct return
    if pool is None: return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    # if norm
    if norm: x = norm(x)
    # return
    return x


'''DropPath'''
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    '''forward'''
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


'''MLP'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_cfg={'type': 'ReLU'}, sigmoid_output=False):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = BuildActivation(act_cfg=act_cfg)
    '''forward'''
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


'''PatchEmbed'''
class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3), in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    '''forward'''
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


'''MultiScaleAttention'''
class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super(MultiScaleAttention, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)
    '''forward'''
    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = dopool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1: 3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        x = self.proj(x)
        # return
        return x


'''MultiScaleBlock'''
class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.0, norm_cfg={'type': 'LayerNorm', 'eps': 1e-6}, q_stride=None, act_cfg={'type': 'GELU'}, window_size=0):
        super(MultiScaleBlock, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = BuildNormalization(dim, norm_cfg)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = BuildNormalization(dim_out, norm_cfg)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, act_cfg=act_cfg)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
    '''forward'''
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # skip connection
        if self.dim != self.dim_out:
            shortcut = dopool(self.proj(x), self.pool)
        # window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = windowpartition(x, window_size)
        # window attention + q pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)
        # reverse window partition
        if self.window_size > 0:
            x = windowunpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        # mlp
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # return
        return x


'''Hiera'''
class Hiera(nn.Module):
    def __init__(
        self, embed_dim=96, num_heads=1, drop_path_rate=0.0, q_pool=3, q_stride=(2, 2), stages=(2, 3, 16, 3), dim_mul=2.0, head_mul=2.0, 
        window_pos_embed_bkg_spatial_size=(14, 14), window_spec=(8, 4, 14, 7), global_att_blocks=(12, 16, 20), return_interm_layers=True
    ):
        super(Hiera, self).__init__()
        # assert
        assert len(stages) == len(window_spec)
        # set attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.window_spec = window_spec
        self.global_att_blocks = global_att_blocks
        self.return_interm_layers = return_interm_layers
        # build layers
        depth = sum(stages)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads, drop_path=dpr[i], q_stride=self.q_stride if i in self.q_pool_blocks else None, window_size=window_size,
            )
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]] if return_interm_layers else [self.blocks[-1].dim_out]
        )
    '''getposembed'''
    def getposembed(self, hw):
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed
    '''forward'''
    def forward(self, x):
        # x: (B, H, W, C)
        x = self.patch_embed(x)
        # add pos embed
        x = x + self.getposembed(x.shape[1:3])
        # get outputs
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        # return
        return outputs


'''FPNNeck'''
class FPNNeck(nn.Module):
    def __init__(self, position_encoding_cfg, d_model, backbone_channel_list, kernel_size=1, stride=1, padding=0, fpn_interp_model="bilinear", fuse_type="sum", fpn_top_down_levels=None):
        super(FPNNeck, self).__init__()
        # assert
        assert fuse_type in ['sum', 'avg']
        # set attributes
        self.fuse_type = fuse_type
        self.fpn_interp_model = fpn_interp_model
        self.position_encoding = BuildPE(position_encoding_cfg)
        self.backbone_channel_list = backbone_channel_list
        # build layers
        self.convs = nn.ModuleList()
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module("conv", nn.Conv2d(in_channels=dim, out_channels=d_model, kernel_size=kernel_size, stride=stride, padding=padding))
            self.convs.append(current)
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)
    '''forward'''
    def forward(self, xs):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32), scale_factor=2.0, mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False), antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)
        # return
        return out, pos


'''HieraWithFPN'''
class HieraWithFPN(nn.Module):
    def __init__(self, hiera_cfg, fpn_cfg, scalp=0):
        super(HieraWithFPN, self).__init__()
        self.trunk = Hiera(**hiera_cfg)
        self.neck = FPNNeck(**fpn_cfg)
        self.scalp = int(scalp)
        assert self.trunk.channel_list == self.neck.backbone_channel_list, f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
    '''forward'''
    def forward(self, sample):
        # forward through backbone
        features, pos = self.neck(self.trunk(sample))
        # discard the lowest resolution features
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        # construct output
        src = features[-1]
        output = {
            "vision_features": src, "vision_pos_enc": pos, "backbone_fpn": features,
        }
        # return
        return output