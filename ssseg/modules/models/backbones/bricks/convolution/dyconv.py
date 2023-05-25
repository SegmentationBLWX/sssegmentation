'''
Function:
    Implementation of DynamicConv2d
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..activation import BuildActivation
from ..normalization import BuildNormalization


'''Attention2d'''
class Attention2d(nn.Module):
    def __init__(self, in_channels, out_channels, temperature):
        super(Attention2d, self).__init__()
        assert temperature % 3 == 1
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, out_channels, kernel_size=1, stride=1, padding=0),
        )
    '''update'''
    def update(self):
        if self.temperature != 1: self.temperature -= 3
    '''forward'''
    def forward(self, x):
        x = self.avgpool(x)
        x = self.convs(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


'''DynamicConv2d'''
class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, norm_cfg=None, act_cfg=None):
        super(DynamicConv2d, self).__init__()
        assert in_channels % groups == 0
        # set attrs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = None
        self.K = K
        self.temperature = temperature
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # define modules
        self.attention = Attention2d(in_channels, K, temperature)
        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_channels))
        if norm_cfg is not None: 
            self.norm = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        if act_cfg is not None: 
            self.activation = BuildActivation(act_cfg)
    '''update'''
    def update(self):
        self.attention.update()
    '''forward'''
    def forward(self, x):
        batch_size, in_channels, h, w = x.size()
        softmax_attention = self.attention(x)
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight)
        aggregate_weight = aggregate_weight.view(-1, in_channels, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(
                input=x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )
        else:
            output = F.conv2d(
                input=x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        if hasattr(self, 'norm'): output = self.norm(output)
        if hasattr(self, 'activation'): output = self.activation(output)
        return output