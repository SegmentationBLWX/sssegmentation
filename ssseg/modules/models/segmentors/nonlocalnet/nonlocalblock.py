'''
Function:
    Implementation of NonLocal2d, NonLocal3d, NonLocal3d
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from abc import ABCMeta
from ...backbones import BuildActivation, BuildNormalization


'''_NonLocalNd'''
class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels, reduction=2, use_scale=True, mode='embeddedgaussian', norm_cfg=None, act_cfg=None):
        super(_NonLocalNd, self).__init__()
        assert mode in ['gaussian', 'embeddedgaussian', 'dotproduct', 'concatenation']
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        self.g = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=self.in_channels, norm_cfg=norm_cfg),
        )
        if self.mode != 'gaussian':
            self.theta = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            )
            self.phi = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            )
        if self.mode == 'concatenation':
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1, stride=1, padding=0, bias=False),
                BuildActivation(act_cfg),
            )
    '''gaussian'''
    def gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    '''embedded gaussian'''
    def embeddedgaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    '''dot product'''
    def dotproduct(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight
    '''concatenation'''
    def concatenation(self, theta_x, phi_x):
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight
    '''forward'''
    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *x.size()[2:])
        output = x + self.conv_out(y)
        return output
    

'''NonLocal1d'''
class NonLocal1d(_NonLocalNd):
    def __init__(self, in_channels, sub_sample=False, **kwargs):
        super(NonLocal1d, self).__init__(in_channels, **kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


'''NonLocal2d'''
class NonLocal2d(_NonLocalNd):
    def __init__(self, in_channels, sub_sample=False, **kwargs):
        super(NonLocal2d, self).__init__(in_channels, **kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


'''NonLocal3d'''
class NonLocal3d(_NonLocalNd):
    def __init__(self, in_channels, sub_sample=False, **kwargs):
        super(NonLocal3d, self).__init__(in_channels, **kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer