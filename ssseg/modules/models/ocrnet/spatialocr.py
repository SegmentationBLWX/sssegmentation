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
    def __init__(self, in_channels, key_channels, out_channels, pool_size=1, **kwargs):
        super(SpatialOCRModule, self).__init__()
        align_corners = kwargs.get('align_corners', True)
        normlayer_opts = kwargs.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = kwargs.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        ocb_args = {
            'pool_size': pool_size,
            'in_channels': in_channels,
            'key_channels': key_channels,
            'align_corners': align_corners,
            'normlayer_opts': normlayer_opts,
            'activation_opts': activation_opts,
        }
        self.object_context_block = ObjectContextBlock(**ocb_args)
        self.conv_bn_act = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                         BuildNormalizationLayer(normlayer_opts['type'], (out_channels, normlayer_opts['opts'])),
                                         BuildActivation(activation_opts['type'], **activation_opts['opts']))
    def forward(self, x, proxy_feats):
        context = self.object_context_block(x, proxy_feats)
        output = self.conv_bn_act(torch.cat([context, x], dim=1))
        return output