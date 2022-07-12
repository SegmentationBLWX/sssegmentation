'''
Function:
    Define the Squeeze-and-Excitation Module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..misc import makedivisible
from ..activation import BuildActivation


'''SqueezeExcitationConv2d'''
class SqueezeExcitationConv2d(nn.Module):
    def __init__(self, channels, ratio=16, act_cfgs=None):
        super(SqueezeExcitationConv2d, self).__init__()
        assert act_cfgs is not None, 'argument act_cfgs should be given'
        assert len(act_cfgs) == 2, 'length of act_cfgs should be equal to 2'
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        act_cfg = act_cfgs[0]
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(channels, makedivisible(channels//ratio, 8), kernel_size=1, stride=1, padding=0))
        self.conv1.add_module('activation', BuildActivation(act_cfg))
        act_cfg = act_cfgs[1]
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv', nn.Conv2d(makedivisible(channels//ratio, 8), channels, kernel_size=1, stride=1, padding=0))
        self.conv2.add_module('activation', BuildActivation(act_cfg))
    '''forward'''
    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out