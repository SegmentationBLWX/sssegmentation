'''
Function:
    Implementation of ObjectContextBlock
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''ObjectContextBlock'''
class ObjectContextBlock(SelfAttentionBlock):
    def __init__(self, in_channels, transform_channels, scale, align_corners=False, norm_cfg=None, act_cfg=None):
        self.align_corners = align_corners
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectContextBlock, self).__init__(
            key_in_channels=in_channels, query_in_channels=in_channels, transform_channels=transform_channels, out_channels=in_channels, share_key_query=False,
            query_downsample=query_downsample, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, 
            matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, query_feats, key_feats):
        h, w = query_feats.size()[2:]
        context = super(ObjectContextBlock, self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return output