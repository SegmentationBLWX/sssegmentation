'''
Function:
    Implementation of L2Norm
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''L2Norm'''
class L2Norm(nn.Module):
    def __init__(self, channels, scale=10, eps=1e-10):
        super(L2Norm, self).__init__()
        self.channels, self.eps = channels, eps
        self.weight = nn.Parameter(torch.Tensor(channels))
        nn.init.constant_(self.weight, scale)
    '''forward'''
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out