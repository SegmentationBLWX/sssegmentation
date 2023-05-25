'''
Function:
    Implementation of ContextBlock
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ...backbones import BuildActivation, BuildNormalization


'''ContextBlock'''
class ContextBlock(nn.Module):
    def __init__(self, in_channels, ratio, pooling_type='att', fusion_types=('channel_add', ), norm_cfg=None, act_cfg=None):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1, stride=1, padding=0),
                BuildNormalization(placeholder=[self.planes, 1, 1], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1, stride=1, padding=0),
                BuildNormalization(placeholder=[self.planes, 1, 1], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.channel_mul_conv = None
    '''spatial pool'''
    def spatialpool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context
    '''forward'''
    def forward(self, x):
        context = self.spatialpool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out