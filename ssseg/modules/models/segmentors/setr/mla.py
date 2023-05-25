'''
Function:
    Implementation of Multi-level Feature Aggregation
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ...backbones import BuildNormalization, BuildActivation


'''MLAModule'''
class MLAModule(nn.Module):
    def __init__(self, in_channels_list=[1024, 1024, 1024, 1024], out_channels=256, norm_cfg=None, act_cfg=None):
        super(MLAModule, self).__init__()
        self.channel_proj = nn.ModuleList()
        for i in range(len(in_channels_list)):
            self.channel_proj.append(nn.Sequential(
                nn.Conv2d(in_channels_list[i], out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            ))
        self.feat_extract = nn.ModuleList()
        for i in range(len(in_channels_list)):
            self.feat_extract.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            ))
    '''forward'''
    def forward(self, inputs):
        # feat_list -> [p2, p3, p4, p5]
        feat_list = []
        for x, conv in zip(inputs, self.channel_proj):
            feat_list.append(conv(x))
        # feat_list -> [p5, p4, p3, p2], mid_list -> [m5, m4, m3, m2]
        feat_list = feat_list[::-1]
        mid_list = []
        for feat in feat_list:
            if len(mid_list) == 0: mid_list.append(feat)
            else: mid_list.append(mid_list[-1] + feat)
        # mid_list -> [m5, m4, m3, m2], out_list -> [o2, o3, o4, o5]
        out_list = []
        for mid, conv in zip(mid_list, self.feat_extract):
            out_list.append(conv(mid))
        # return
        return tuple(out_list)


'''MLANeck'''
class MLANeck(nn.Module):
    def __init__(self, in_channels_list, out_channels, norm_layers, norm_cfg=None, act_cfg=None):
        super(MLANeck, self).__init__()
        assert isinstance(in_channels_list, list) or isinstance(in_channels_list, tuple)
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.norm_layers = norm_layers
        self.mla = MLAModule(in_channels_list, out_channels, norm_cfg, act_cfg)
    '''forward'''
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels_list)
        outs = []
        for i in range(len(inputs)):
            x = inputs[i]
            n, c, h, w = x.shape
            x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
            x = self.norm_layers[i](x)
            x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
            outs.append(x)
        outs = self.mla(outs)
        return tuple(outs)