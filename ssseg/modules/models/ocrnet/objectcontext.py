'''
Function:
    define the object context block
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalizationLayer


'''object context block'''
class ObjectContextBlock(nn.Module):
    def __init__(self, in_channels, key_channels, pool_size=1, **kwargs):
        super(ObjectContextBlock, self).__init__()
        # parse and set args
        align_corners = kwargs.get('align_corners', True)
        normlayer_opts = kwargs.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = kwargs.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool_size = pool_size
        # define the module layers
        self.maxpool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
        self.f_pixel = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                     BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                     BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                     nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                     BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                     BuildActivation(activation_opts['type'], **activation_opts['opts']),)
        self.f_object = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                      BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                      nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                      BuildActivation(activation_opts['type'], **activation_opts['opts']),)
        self.f_down = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                    BuildNormalizationLayer(normlayer_opts['type'], (key_channels, normlayer_opts['opts'])),
                                    BuildActivation(activation_opts['type'], **activation_opts['opts']))
        self.f_up = nn.Sequential(nn.Conv2d(key_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                  BuildNormalizationLayer(normlayer_opts['type'], (in_channels, normlayer_opts['opts'])),
                                  BuildActivation(activation_opts['type'], **activation_opts['opts']))
    def forward(self, x, proxy_feats):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.pool_size > 1: x = self.maxpool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy_feats).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy_feats).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.pool_size > 1: context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context