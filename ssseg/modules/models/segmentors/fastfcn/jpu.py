'''
Function:
    Implementation of Joint Pyramid Upsampling (JPU)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalization, DepthwiseSeparableConv2d


'''JPU'''
class JPU(nn.Module):
    def __init__(self, in_channels_list=(512, 1024, 2048), mid_channels=512, start_level=0, end_level=-1, dilations=(1, 2, 4, 8), 
                 align_corners=False, norm_cfg=None, act_cfg=None):
        super(JPU, self).__init__()
        # set attrs
        self.in_channels_list = in_channels_list
        self.mid_channels = mid_channels
        self.start_level = start_level
        self.num_ins = len(in_channels_list)
        if end_level == -1: 
            self.backbone_end_level = self.num_ins
        else: 
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels_list)
        self.dilations = dilations
        self.align_corners = align_corners
        # define modules
        self.conv_layers = nn.ModuleList()
        self.dilation_layers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            conv_layer = nn.Sequential(
                nn.Conv2d(self.in_channels_list[i], self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=self.mid_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            self.conv_layers.append(conv_layer)
        for idx in range(len(dilations)):
            dilation_layer = DepthwiseSeparableConv2d(
                in_channels=(self.backbone_end_level - self.start_level) * self.mid_channels, out_channels=self.mid_channels,
                kernel_size=3, stride=1, padding=dilations[idx], dilation=dilations[idx], dw_norm_cfg=norm_cfg, dw_act_cfg=None,
                pw_norm_cfg=norm_cfg, pw_act_cfg=act_cfg,
            )
            self.dilation_layers.append(dilation_layer)
    '''forward'''
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels_list)
        feats = [
            self.conv_layers[idx - self.start_level](inputs[idx]) for idx in range(self.start_level, self.backbone_end_level)
        ]
        h, w = feats[0].shape[2:]
        for idx in range(1, len(feats)):
            feats[idx] = F.interpolate(feats[idx], size=(h, w), mode='bilinear', align_corners=self.align_corners)
        feat = torch.cat(feats, dim=1)
        concat_feat = torch.cat([
            self.dilation_layers[idx](feat) for idx in range(len(self.dilations))
        ], dim=1)
        outs = []
        for i in range(self.start_level, self.backbone_end_level - 1):
            outs.append(inputs[i])
        outs.append(concat_feat)
        return tuple(outs)