'''
Function:
    Implementation of Disentangled Non-Local Block
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..nonlocalnet import NonLocal2d


'''DisentangledNonLocal2d'''
class DisentangledNonLocal2d(NonLocal2d):
    def __init__(self, *arg, temperature, **kwargs):
        super(DisentangledNonLocal2d, self).__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=1, padding=0)
    '''embedded gaussian with temperature'''
    def embeddedgaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight /= self.temperature
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    '''forward'''
    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample: phi_x = self.phi(x).view(n, self.in_channels, -1)
            else: phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
        theta_x = theta_x - theta_x.mean(dim=-2, keepdim=True)
        phi_x = phi_x - phi_x.mean(dim=-1, keepdim=True)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *x.size()[2:])
        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.view(n, 1, -1)
        unary_mask = unary_mask.softmax(dim=-1)
        unary_x = torch.matmul(unary_mask, g_x)
        unary_x = unary_x.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, 1, 1)
        output = x + self.conv_out(y + unary_x)
        return output