'''
Function:
    define the edge perceiving module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalizationLayer


'''edge perceiving module'''
class EdgePerceivingModule(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024], hidden_channels=256, out_channels=2, **kwargs):
        super(EdgePerceivingModule, self).__init__()
        align_corners = kwargs.get('align_corners', True)
        normlayer_opts = kwargs.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = kwargs.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        self.align_corners = align_corners
        self.branches = nn.ModuleList()
        for in_channels in in_channels_list:
            self.branches.append(nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                               BuildNormalizationLayer(normlayer_opts['type'], (hidden_channels, normlayer_opts['opts'])),
                                               BuildActivation(activation_opts['type'], **activation_opts['opts'])))
        self.edge_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.fuse_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        assert len(x) == len(self.branches)
        h, w = x[0].size(2), x[0].size(3)
        edges_feats, edges = [], []
        for i in range(len(x)):
            edge_feats = self.branches[i](x[i])
            edge = self.edge_conv(edge_feats)
            if i > 0:
                edge_feats = F.interpolate(edge_feats, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            edges_feats.append(edge_feats)
            edges.append(edge)
        edge_feats = torch.cat(edges_feats, dim=1)
        edge = torch.cat(edges, dim=1)
        edge = self.fuse_conv(edge)
        return edge, edge_feats