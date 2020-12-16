'''
Function:
    define the spatial ocr module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .objectcontext import ObjectContextBlock
from ...backbones import BuildActivation, BuildNormalizationLayer


'''spatial ocr module'''
class SpatialOCRModule(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, **kwargs):
        super(SpatialOCRModule, self).__init__()
        align_corners, norm_cfg, act_cfg = kwargs['align_corners'], kwargs['norm_cfg'], kwargs['act_cfg']
        ocb_args = {
            'in_channels': in_channels,
            'key_channels': key_channels,
            'align_corners': align_corners,
            'norm_cfg': norm_cfg,
            'act_cfg': act_cfg,
        }
        self.object_context_block = ObjectContextBlock(**ocb_args)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalizationLayer(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
    '''forward'''
    def forward(self, x, proxy_feats):
        context = self.object_context_block(x, proxy_feats)
        output = self.bottleneck(torch.cat([context, x], dim=1))
        return output