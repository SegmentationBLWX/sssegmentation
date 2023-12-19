'''
Function:
    Implementation of AdptivePaddingConv2d
Author:
    Zhenchao Jin
'''
import math
import torch.nn as nn
import torch.nn.functional as F
from ..normalization import BuildNormalization


'''AdptivePaddingConv2d'''
class AdptivePaddingConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm_cfg=None, act_cfg=None):
        super(AdptivePaddingConv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        if norm_cfg is not None: 
            self.norm = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        if act_cfg is not None: 
            self.activation = BuildActivation(act_cfg)
    '''forward'''
    def forward(self, x):
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if hasattr(self, 'norm'): output = self.norm(output)
        if hasattr(self, 'activation'): output = self.activation(output)
        return output