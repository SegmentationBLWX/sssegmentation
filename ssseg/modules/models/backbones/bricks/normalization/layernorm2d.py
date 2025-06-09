'''
Function:
    Implementation of LayerNorm2d
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''LayerNorm2d'''
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, **kwargs):
        super(LayerNorm2d, self).__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]
    '''forward'''
    def forward(self, x: torch.Tensor, data_format='channel_first'):
        assert x.dim() == 4, f'LayerNorm2d only supports inputs with shape (N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x