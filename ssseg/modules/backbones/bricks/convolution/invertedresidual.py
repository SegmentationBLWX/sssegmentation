'''
Function:
    define InvertedResidual Layer
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..activation import BuildActivation
from ..normalization import BuildNormalizationLayer


'''InvertedResidual'''
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_cfg=None, act_cfg=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s...' % stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layer = nn.Sequential()
            layer.add_module('conv', nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layer.add_module('bn', BuildNormalizationLayer(norm_cfg['type'], (hidden_dim, norm_cfg['opts'])))
            layer.add_module('activation', BuildActivation(act_cfg['type'], **act_cfg['opts']))
            layers.append(layer)
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_dim, bias=False))
        layer.add_module('bn', BuildNormalizationLayer(norm_cfg['type'], (hidden_dim, norm_cfg['opts'])))
        layer.add_module('activation', BuildActivation(act_cfg['type'], **act_cfg['opts']))
        layers.extend([layer])
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layer.add_module('bn', BuildNormalizationLayer(norm_cfg['type'], (out_channels, norm_cfg['opts'])))
        layers.extend([layer])
        self.conv = nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)