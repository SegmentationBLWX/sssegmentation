'''
Function:
    Implementation of Scale
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''Scale'''
class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    '''forward'''
    def forward(self, x):
        return x * self.scale


'''LayerScale'''
class LayerScale(nn.Module):
    def __init__(self, dim: int, inplace: bool = False, data_format: str = 'channels_last', scale: float = 1e-5):
        super(LayerScale, self).__init__()
        assert data_format in ('channels_last', 'channels_first'), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)
    '''forward'''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == 'channels_first':
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)