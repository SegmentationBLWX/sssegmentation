'''
Function:
    Image to Patch Embedding
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..normalization import BuildNormalization


'''Image to Patch Embedding'''
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dims=768, kernel_size=16, stride=16, padding=0, dilation=1, pad_to_patch_size=True, norm_cfg=None):
        super(PatchEmbed, self).__init__()
        self.embed_dims = embed_dims
        if stride is None: stride = kernel_size
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.pad_to_patch_size = pad_to_patch_size
        if norm_cfg is not None:
            self.norm = BuildNormalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))
        else:
            self.norm = None
    '''forward'''
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        if self.pad_to_patch_size:
            if H % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))
        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x