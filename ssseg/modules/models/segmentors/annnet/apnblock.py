'''
Function:
    Implementation of Asymmetric Pyramid Non-local Block (APNB)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .ppm import PPMConcat
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''Asymmetric Pyramid Non-local Block (APNB)'''
class APNBlock(nn.Module):
    def __init__(self, in_channels, transform_channels, out_channels, query_scales, key_pool_scales, norm_cfg=None, act_cfg=None):
        super(APNBlock, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            key_psp = PPMConcat(key_pool_scales)
            if query_scale > 1:
                query_downsample = nn.MaxPool2d(kernel_size=query_scale)
            else:
                query_downsample = None
            self.stages.append(SelfAttentionBlock(
                key_in_channels=in_channels, query_in_channels=in_channels, transform_channels=transform_channels, out_channels=out_channels, 
                share_key_query=True, query_downsample=query_downsample, key_downsample=key_psp, key_query_num_convs=1, value_out_num_convs=1,
                key_query_norm=True, value_out_norm=False, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, feats):
        priors = [stage(feats, feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, feats], 1))
        return output