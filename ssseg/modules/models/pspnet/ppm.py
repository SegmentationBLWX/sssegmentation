'''
Function:
    define the Pyramid Pooling Module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalizationLayer


'''Pyramid Pooling Module'''
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes, **kwargs):
        super(PyramidPoolingModule, self).__init__()
        align_corners, norm_cfg, act_cfg = kwargs['align_corners'], kwargs['norm_cfg'], kwargs['act_cfg']
        self.align_corners = align_corners
        self.branches = nn.ModuleList()
        for bin_size in bin_sizes:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalizationLayer(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts'])
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(bin_sizes), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    '''forward'''
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_lvls = [x]
        for branch in self.branches:
            out = branch(x)
            pyramid_lvls.append(F.interpolate(out, size=(h, w), mode='bilinear', align_corners=self.align_corners))
        output = torch.cat(pyramid_lvls, dim=1)
        output = self.bottleneck(output)
        return output