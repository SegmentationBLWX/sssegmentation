'''
Function:
    Implementation of GRN (Global Response Normalization)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''GRN'''
class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    '''forward'''
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x