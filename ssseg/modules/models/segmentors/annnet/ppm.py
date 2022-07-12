'''
Function:
    Implementation of Pyramid Pooling Module (Concat only)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''Pyramid Pooling Module (Concat only)'''
class PPMConcat(nn.Module):
    def __init__(self, pool_scales):
        super(PPMConcat, self).__init__()
        self.pool_nets = nn.ModuleList()
        for pool_scale in pool_scales:
            self.pool_nets.append(nn.AdaptiveAvgPool2d(pool_scale))
    '''forward'''
    def forward(self, x):
        ppm_outs = []
        for pool_net in self.pool_nets:
            ppm_out = pool_net(x)
            ppm_outs.append(ppm_out.view(*x.shape[:2], -1))
        ppm_outs = torch.cat(ppm_outs, dim=2)
        return ppm_outs