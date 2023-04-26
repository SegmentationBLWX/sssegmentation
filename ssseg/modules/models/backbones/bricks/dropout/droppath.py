'''
Function:
    Implementation of DropPath
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''DropPath'''
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
    '''forward'''
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(self.keep_prob) * random_tensor
        return output