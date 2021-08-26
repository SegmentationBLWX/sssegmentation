'''
Function:
    define some utils used in transformer backbones
Author:
    Zhenchao Jin
'''
import torch


'''Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor'''
def nlctonchw(x, hw_shape):
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


'''Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor'''
def nchwtonlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()