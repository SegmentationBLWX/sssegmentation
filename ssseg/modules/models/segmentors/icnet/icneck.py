'''
Function:
    Implementation of ICNeck
Author:
    Zhenchao Jin
'''
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildNormalization, BuildActivation


'''CascadeFeatureFusion'''
class CascadeFeatureFusion(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, norm_cfg=None, act_cfg=None, align_corners=False):
        super(CascadeFeatureFusion, self).__init__()
        self.align_corners = align_corners
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = F.relu(x_low + x_high, inplace=True)
        return x, x_low


'''ICNeck'''
class ICNeck(nn.Module):
    def __init__(self, in_channels_list=(64, 256, 256), out_channels=128, norm_cfg=None, act_cfg=None, align_corners=False):
        super(ICNeck, self).__init__()
        assert len(in_channels_list) == 3, 'in_channels_list should be equal to 3'
        # set attrs
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        # define modules
        self.cff_24 = CascadeFeatureFusion(
            low_channels=in_channels_list[2], high_channels=in_channels_list[1], out_channels=out_channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg, align_corners=align_corners,
        )
        self.cff_12 = CascadeFeatureFusion(
            low_channels=out_channels, high_channels=in_channels_list[0], out_channels=out_channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg, align_corners=align_corners,
        )
    '''forward'''
    def forward(self, inputs):
        assert len(inputs) == 3, 'length of input feature maps must be 3'
        x_sub1, x_sub2, x_sub4 = inputs
        x_cff_24, x_24 = self.cff_24(x_sub4, x_sub2)
        x_cff_12, x_12 = self.cff_12(x_cff_24, x_sub1)
        return x_24, x_12, x_cff_12