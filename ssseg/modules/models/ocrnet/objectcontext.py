'''
Function:
    define the object context block
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.selfattention import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalizationLayer


'''object context block'''
class ObjectContextBlock(nn.Module):
    def __init__(self, in_channels, key_channels, **kwargs):
        super(ObjectContextBlock, self).__init__()
        # parse and set args
        align_corners, norm_cfg, act_cfg = kwargs['align_corners'], kwargs['norm_cfg'], kwargs['act_cfg']
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.key_channels = key_channels
        # define the module layers
        query_project = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (key_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (key_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        key_project = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (key_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (key_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        value_project = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (key_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        out_project = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (in_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        self_attention_cfg = {
            'matmul_norm_cfg': {'is_on': True, 'key_channels': key_channels},
            'query_downsample': None,
            'key_downsample': None,
            'query_project': query_project,
            'key_project': key_project,
            'value_project': value_project,
            'out_project': out_project
        }
        self.self_attention_net = SelfAttentionBlock(**self_attention_cfg)
    '''forward'''
    def forward(self, x, proxy_feats):
        return self.self_attention_net(x, proxy_feats)