'''
Function:
    Implementation of MemoryEncoder
Author:
    Zhenchao Jin
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation
from ...backbones.samvit import LayerNorm2d
from ...backbones.hiera import DropPath, BuildPE
from ..mask2former.transformers.misc import getclones


'''MaskDownSampler'''
class MaskDownSampler(nn.Module):
    def __init__(self, embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16, act_cfg={'type': 'GELU'}):
        super(MaskDownSampler, self).__init__()
        # assert
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        # set layers
        self.encoder = []
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(nn.Conv2d(mask_in_chans, mask_out_chans, kernel_size=kernel_size, stride=stride, padding=padding))
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(BuildActivation(act_cfg=act_cfg))
            mask_in_chans = mask_out_chans
        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.encoder = nn.Sequential(*self.encoder)
    '''forward'''
    def forward(self, x):
        return self.encoder(x)


'''CXBlock'''
class CXBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3, drop_path=0.0, layer_scale_init_value=1e-6, use_dwconv=True):
        super(CXBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim if use_dwconv else 1)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    '''forward'''
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


'''Fuser'''
class Fuser(nn.Module):
    def __init__(self, layer_cfg, num_layers, dim=None, input_projection=False):
        super(Fuser, self).__init__()
        self.proj = nn.Identity()
        layer = CXBlock(**layer_cfg)
        self.layers = getclones(layer, num_layers)
        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)
    '''forward'''
    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


'''MemoryEncoder'''
class MemoryEncoder(nn.Module):
    def __init__(self, out_dim, mask_downsampler_cfg, fuser_cfg, position_encoding_cfg, in_dim=256):
        super(MemoryEncoder, self).__init__()
        self.mask_downsampler = MaskDownSampler(**mask_downsampler_cfg)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(**fuser_cfg)
        self.position_encoding = BuildPE(position_encoding_cfg)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    '''forward'''
    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        # Process masks: sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        # Fuse pix_feats and downsampled masks: in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x).to(x.dtype)
        # return
        return {"vision_features": x, "vision_pos_enc": [pos]}