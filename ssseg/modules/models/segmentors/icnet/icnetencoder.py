'''
Function:
    Implementation of ICNetEncoder
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.nn.functional as F
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildNormalization, BuildActivation, BuildBackbone


'''ICNetEncoder'''
class ICNetEncoder(nn.Module):
    def __init__(self, backbone_cfg=None, in_channels=3, layer_channels_list=(512, 2048), light_branch_middle_channels=32, psp_out_channels=512, 
                 out_channels_list=(64, 256, 256), pool_scales=(1, 2, 3, 6), norm_cfg=None, act_cfg=None, align_corners=False):
        super(ICNetEncoder, self).__init__()
        self.align_corners = align_corners
        assert (backbone_cfg is not None) and isinstance(backbone_cfg, dict)
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
        self.backbone_net.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.ppm_net = PyramidPoolingModule(
            pool_scales=pool_scales, in_channels=layer_channels_list[1], out_channels=psp_out_channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg, align_corners=align_corners,
        )
        self.conv_sub1 = nn.Sequential(
            nn.Conv2d(in_channels, light_branch_middle_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=light_branch_middle_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(light_branch_middle_channels, light_branch_middle_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=light_branch_middle_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(light_branch_middle_channels, out_channels_list[0], kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels_list[0], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.conv_sub2 = nn.Sequential(
            nn.Conv2d(layer_channels_list[0], out_channels_list[1], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels_list[1], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.conv_sub4 = nn.Sequential(
            nn.Conv2d(psp_out_channels, out_channels_list[2], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels_list[2], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        output = []
        # sub 1
        output.append(self.conv_sub1(x))
        # sub 2
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        x = self.backbone_net.stem(x)
        x = self.backbone_net.maxpool(x)
        x = self.backbone_net.layer1(x)
        x = self.backbone_net.layer2(x)
        output.append(self.conv_sub2(x))
        # sub 4
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        x = self.backbone_net.layer3(x)
        x = self.backbone_net.layer4(x)
        ppm_out = self.ppm_net(x)
        output.append(self.conv_sub4(ppm_out))
        # return
        return output