'''
Function:
    Implementation of Pyramid Pooling Module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalization


'''PyramidPoolingModule'''
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales, align_corners=False, norm_cfg=None, act_cfg=None):
        super(PyramidPoolingModule, self).__init__()
        self.align_corners = align_corners
        self.branches = nn.ModuleList()
        for pool_scale in pool_scales:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg)
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_scales), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg)
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