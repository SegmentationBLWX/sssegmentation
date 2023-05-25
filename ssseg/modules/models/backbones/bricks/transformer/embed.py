'''
Function:
    Implementation of AdaptivePadding, PatchEmbed and PatchMerging
Author:
    Zhenchao Jin
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..normalization import BuildNormalization


'''AdaptivePadding'''
class AdaptivePadding(nn.Module):
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        self.padding = padding
        self.kernel_size = self.totuple(kernel_size)
        self.stride = self.totuple(stride)
        self.dilation = self.totuple(dilation)
    '''get pad shape'''
    def getpadshape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w
    '''forward'''
    def forward(self, x):
        pad_h, pad_w = self.getpadshape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x
    '''to tuple'''
    @staticmethod
    def totuple(x):
        if isinstance(x, int): return (x, x)
        assert isinstance(x, tuple) and (len(x) == 2)
        return x


'''PatchEmbed'''
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dims=768, kernel_size=16, stride=None, padding='corner', dilation=1, bias=True, norm_cfg=None, input_size=None):
        super(PatchEmbed, self).__init__()
        # initialize
        self.embed_dims = embed_dims
        if stride is None: stride = kernel_size
        stride = AdaptivePadding.totuple(stride)
        dilation = AdaptivePadding.totuple(dilation)
        kernel_size = AdaptivePadding.totuple(kernel_size)
        # adaptive padding
        self.adap_padding = None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        # projection
        padding = AdaptivePadding.totuple(padding)
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        # norm
        self.norm = None
        if norm_cfg is not None: 
            self.norm = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        # input size
        self.init_input_size = None
        self.init_out_size = None
        if input_size:
            input_size = AdaptivePadding.totuple(input_size)
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.getpadshape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
    '''forward'''
    def forward(self, x):
        if self.adap_padding: x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None: x = self.norm(x)
        return x, out_size


'''PatchMerging'''
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=None, padding='corner', dilation=1, bias=False, norm_cfg=None):
        super(PatchMerging, self).__init__()
        # initialize
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride: stride = stride
        else: stride = kernel_size
        stride = AdaptivePadding.totuple(stride)
        dilation = AdaptivePadding.totuple(dilation)
        kernel_size = AdaptivePadding.totuple(kernel_size)
        # adaptive padding
        self.adap_padding = None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        # sampler
        padding = AdaptivePadding.totuple(padding)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        # norm
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.norm = None
        if norm_cfg is not None: 
            self.norm = BuildNormalization(placeholder=sample_dim, norm_cfg=norm_cfg)
        # reduction
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)
    '''forward'''
    def forward(self, x, input_size):
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]
        x = self.sampler(x)
        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1) // self.sampler.stride[1] + 1
        output_size = (out_h, out_w)
        x = x.transpose(1, 2)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size