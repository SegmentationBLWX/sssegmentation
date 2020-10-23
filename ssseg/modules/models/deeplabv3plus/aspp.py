'''
Function:
    define the Atrous Spatial Pyramid Pooling (ASPP)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, DepthwiseSeparableConv2d, BuildNormalizationLayer


'''ASPP'''
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates, **kwargs):
        super(ASPP, self).__init__()
        align_corners = kwargs.get('align_corners', True)
        normlayer_opts = kwargs.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = kwargs.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, rate in enumerate(rates):
            if rate == 1:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=rate, bias=False),
                                       BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                       BuildActivation(activation_opts['type'], **activation_opts['opts']))
            else:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                                       BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                       BuildActivation(activation_opts['type'], **activation_opts['opts']))
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                           BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                           BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self.fuse_layer = nn.Sequential(nn.Conv2d(out_channels * (len(rates) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                        BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                        BuildActivation(activation_opts['type'], **activation_opts['opts']))
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
        features = self.fuse_layer(features)
        return features


'''Depthwise Separable ASPP'''
class DepthwiseSeparableASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates, **kwargs):
        super(DepthwiseSeparableASPP, self).__init__()
        align_corners = kwargs.get('align_corners', True)
        normlayer_opts = kwargs.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = kwargs.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, rate in enumerate(rates):
            if rate == 1:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=rate, bias=False),
                                       BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                       BuildActivation(activation_opts['type'], **activation_opts['opts']))
            else:
                branch = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False, normlayer_opts=normlayer_opts, activation_opts=activation_opts)
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                           BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                           BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self.fuse_layer = nn.Sequential(nn.Conv2d(out_channels * (len(rates) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                        BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                        BuildActivation(activation_opts['type'], **activation_opts['opts']))
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
        features = self.fuse_layer(features)
        return features