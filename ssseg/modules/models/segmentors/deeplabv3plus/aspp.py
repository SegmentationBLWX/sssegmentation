'''
Function:
    Implementation of Depthwise Separable Atrous Spatial Pyramid Pooling (ASPP)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, DepthwiseSeparableConv2d, BuildNormalization


'''DepthwiseSeparableASPP'''
class DepthwiseSeparableASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, align_corners=False, norm_cfg=None, act_cfg=None):
        super(DepthwiseSeparableASPP, self).__init__()
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False),
                    BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                )
            else:
                branch = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    '''forward'''
    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features