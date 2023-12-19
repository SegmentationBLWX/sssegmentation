'''
Function:
    Implementation of InvertedResidual and InvertedResidualV3
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from ..activation import BuildActivation
from .apconv import AdptivePaddingConv2d
from .seconv import SqueezeExcitationConv2d
from ..normalization import BuildNormalization


'''InvertedResidual'''
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_cfg=None, act_cfg=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s' % stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layer = nn.Sequential()
            layer.add_module('conv', nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            if norm_cfg is not None:
                layer.add_module('bn', BuildNormalization(placeholder=hidden_dim, norm_cfg=norm_cfg))
            if act_cfg is not None:
                layer.add_module('activation', BuildActivation(act_cfg))
            layers.append(layer)
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_dim, bias=False))
        if norm_cfg is not None:
            layer.add_module('bn', BuildNormalization(placeholder=hidden_dim, norm_cfg=norm_cfg))
        if act_cfg is not None:
            layer.add_module('activation', BuildActivation(act_cfg))
        layers.extend([layer])
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if norm_cfg is not None:
            layer.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        layers.extend([layer])
        self.conv = nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


'''InvertedResidualV3'''
class InvertedResidualV3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, se_cfg=None, with_expand_conv=True, norm_cfg=None, act_cfg=None):
        super(InvertedResidualV3, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s' % stride
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        self.with_expand_conv = with_expand_conv
        if not self.with_expand_conv: assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = nn.Sequential()
            self.expand_conv.add_module('conv', nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            if norm_cfg is not None:
                self.expand_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.expand_conv.add_module('activation', BuildActivation(act_cfg))
        self.depthwise_conv = nn.Sequential()
        if stride == 2:
            self.depthwise_conv.add_module('conv', AdptivePaddingConv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
            if norm_cfg is not None:
                self.depthwise_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.depthwise_conv.add_module('activation', BuildActivation(act_cfg))
        else:
            self.depthwise_conv.add_module('conv', nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
            if norm_cfg is not None:
                self.depthwise_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.depthwise_conv.add_module('activation', BuildActivation(act_cfg))
        if se_cfg is not None:
            self.se = SqueezeExcitationConv2d(**se_cfg)
        self.linear_conv = nn.Sequential()
        self.linear_conv.add_module('conv', nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if norm_cfg is not None:
            self.linear_conv.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
    '''forward'''
    def forward(self, x):
        out = x
        if self.with_expand_conv: out = self.expand_conv(out)
        out = self.depthwise_conv(out)
        if hasattr(self, 'se'): out = self.se(out)
        out = self.linear_conv(out)
        if self.with_res_shortcut:
            return x + out
        return out