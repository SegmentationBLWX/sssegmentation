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
    def __init__(self, drop_prob: float = 0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    '''forward'''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.droppath(x, self.drop_prob, self.training)
    '''droppath'''
    @staticmethod
    def droppath(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        output = x.div(keep_prob) * random_tensor.floor()
        return output