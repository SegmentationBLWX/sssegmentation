'''
Function:
    define the object context block
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .selfattention import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalizationLayer


'''object context block'''
class ObjectContextBlock(nn.Module):
    def __init__(self, in_channels, key_channels, **kwargs):
        super(ObjectContextBlock, self).__init__()
        # parse and set args
        align_corners = kwargs.get('align_corners', True)
        normlayer_opts = kwargs.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = kwargs.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.key_channels = key_channels
        # define the module layers
        self.query_project = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                           BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                           BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                           nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                           BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                           BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self.key_project = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                         BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                         BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                         nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                         BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                         BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self.value_project = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                           BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                           BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self.out_project = nn.Sequential(nn.Conv2d(key_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                         BuildNormalizationLayer(normlayer_opts['type'], (in_channels, normlayer_opts['opts'])),
                                         BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self_attention_cfg = {
            'norm_opts': {'is_on': True, 'key_channels': self.key_channels},
            'query_downsample': None,
            'key_downsample': None,
            'query_project': self.query_project,
            'key_project': self.key_project,
            'value_project': self.value_project,
            'out_project': self.out_project
        }
        self.self_attention_net = SelfAttentionBlock(**self_attention_cfg)
    '''forward'''
    def forward(self, x, proxy_feats):
        return self.self_attention_net(x, proxy_feats)