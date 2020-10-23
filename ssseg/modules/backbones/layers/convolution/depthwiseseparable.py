'''
Function:
    define Depthwise Separable Convolution Layer
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..activation import BuildActivation
from ..normalization import BuildNormalizationLayer


'''Depthwise Separable Conv2d'''
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, normlayer_opts=None, activation_opts=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.depthwise_bn = BuildNormalizationLayer(normlayer_opts['type'], (in_channels, normlayer_opts['opts']))
        self.depthwise_activate = BuildActivation(activation_opts['type'], **activation_opts['opts'])
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts']))
        self.pointwise_activate = BuildActivation(activation_opts['type'], **activation_opts['opts'])
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activate(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_activate(x)
        return x