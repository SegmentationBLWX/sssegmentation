'''
Function:
    Implementation of some utils for transformer backbones
Author:
    Zhenchao Jin
'''
import torch


'''nlctonchw'''
def nlctonchw(x: torch.Tensor, hw_shape):
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


'''nchwtonlc'''
def nchwtonlc(x: torch.Tensor):
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


'''nchw2nlc2nchw'''
def nchw2nlc2nchw(module, x: torch.Tensor, contiguous=False, **kwargs):
    B, C, H, W = x.shape
    if not contiguous:
        x = x.flatten(2).transpose(1, 2)
        x = module(x, **kwargs)
        x = x.transpose(1, 2).reshape(B, C, H, W)
    else:
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = module(x, **kwargs)
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
    return x


'''nlc2nchw2nlc'''
def nlc2nchw2nlc(module, x: torch.Tensor, hw_shape, contiguous=False, **kwargs):
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    if not contiguous:
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = module(x, **kwargs)
        x = x.flatten(2).transpose(1, 2)
    else:
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = module(x, **kwargs)
        x = x.flatten(2).transpose(1, 2).contiguous()
    return x