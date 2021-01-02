'''
Function:
    define Depthwise Separable Convolution Module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..activation import BuildActivation
from ..normalization import BuildNormalization


'''Depthwise Separable Conv2d'''
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_cfg=None, act_cfg=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.depthwise_bn = BuildNormalization(norm_cfg['type'], (in_channels, norm_cfg['opts']))
        self.depthwise_activate = BuildActivation(act_cfg['type'], **act_cfg['opts'])
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.pointwise_bn = BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts']))
        self.pointwise_activate = BuildActivation(act_cfg['type'], **act_cfg['opts'])
    '''forward'''
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activate(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_activate(x)
        return x