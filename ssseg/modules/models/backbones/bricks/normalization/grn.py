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
    def __init__(self, in_channels, eps=1e-6):
        super(GRN, self).__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
    '''forward'''
    def forward(self, x: torch.Tensor, data_format='channel_first'):
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(1, -1, 1, 1) + x
        return x