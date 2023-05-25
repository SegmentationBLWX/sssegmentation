'''
Function:
    Implementation of DepthwiseSeparableConv2d
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..activation import BuildActivation
from ..normalization import BuildNormalization


'''DepthwiseSeparableConv2d'''
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, 
                 norm_cfg=None, act_cfg=None, dw_norm_cfg=None, dw_act_cfg=None, pw_norm_cfg=None, pw_act_cfg=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        if dw_norm_cfg is None: dw_norm_cfg = norm_cfg
        if dw_act_cfg is None: dw_act_cfg = act_cfg
        if pw_norm_cfg is None: pw_norm_cfg = norm_cfg
        if pw_act_cfg is None: pw_act_cfg = act_cfg
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        if dw_norm_cfg is not None:
            self.depthwise_bn = BuildNormalization(placeholder=in_channels, norm_cfg=dw_norm_cfg)
        if dw_act_cfg is not None:
            self.depthwise_activate = BuildActivation(dw_act_cfg)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if pw_norm_cfg is not None:
            self.pointwise_bn = BuildNormalization(placeholder=out_channels, norm_cfg=pw_norm_cfg)
        if pw_act_cfg is not None:
            self.pointwise_activate = BuildActivation(pw_act_cfg)
    '''forward'''
    def forward(self, x):
        x = self.depthwise_conv(x)
        if hasattr(self, 'depthwise_bn'): x = self.depthwise_bn(x)
        if hasattr(self, 'depthwise_activate'): x = self.depthwise_activate(x)
        x = self.pointwise_conv(x)
        if hasattr(self, 'pointwise_bn'): x = self.pointwise_bn(x)
        if hasattr(self, 'pointwise_activate'): x = self.pointwise_activate(x)
        return x