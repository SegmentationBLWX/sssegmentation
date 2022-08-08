'''
Function:
    Implementation of BiSeNetV1
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .bricks import BuildNormalization, BuildActivation, constructnormcfg


'''model urls'''
model_urls = {}


'''Spatial Path to preserve the spatial size of the original input image and encode affluent spatial information'''
class SpatialPath(nn.Module):
    def __init__(self, in_channels=3, num_channels_list=(64, 64, 64, 128), norm_cfg=None, act_cfg=None):
        super(SpatialPath, self).__init__()
        assert len(num_channels_list) == 4
        self.layers = []
        for idx in range(len(num_channels_list)):
            layer_name = f'layer{idx + 1}'
            self.layers.append(layer_name)
            if idx == 0:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, num_channels_list[idx], kernel_size=7, stride=2, padding=3, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=num_channels_list[idx], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg),
                )
            elif idx == len(num_channels_list) - 1:
                conv = nn.Sequential(
                    nn.Conv2d(num_channels_list[idx - 1], num_channels_list[idx], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=num_channels_list[idx], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg),
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(num_channels_list[idx - 1], num_channels_list[idx], kernel_size=3, stride=2, padding=1, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=num_channels_list[idx], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg),
                )
            self.add_module(layer_name, conv)
    '''forward'''
    def forward(self, x):
        for idx, layer_name in enumerate(self.layers):
            layer_stage = getattr(self, layer_name)
            x = layer_stage(x)
        return x


'''Attention Refinement Module (ARM) to refine the features of each stage'''
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(AttentionRefinementModule, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            nn.Sigmoid(),
        )
    '''forward'''
    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


'''Context Path to provide sufficient receptive field'''
class ContextPath(nn.Module):
    def __init__(self, backbone_cfg, context_channels_list=(128, 256, 512), norm_cfg=None, act_cfg=None):
        super(ContextPath, self).__init__()
        assert len(context_channels_list) == 3
        if 'norm_cfg' not in backbone_cfg: backbone_cfg['norm_cfg'] = norm_cfg
        self.backbone_net = self.buildbackbone(backbone_cfg)
        self.arm16 = AttentionRefinementModule(context_channels_list[1], context_channels_list[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.arm32 = AttentionRefinementModule(context_channels_list[2], context_channels_list[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_head32 = nn.Sequential(
            nn.Conv2d(context_channels_list[0], context_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=context_channels_list[0], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.conv_head16 = nn.Sequential(
            nn.Conv2d(context_channels_list[0], context_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=context_channels_list[0], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(context_channels_list[2], context_channels_list[0], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=context_channels_list[0], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone_net(x)
        x_gap = self.gap_conv(x_32)
        x_32_arm = self.arm32(x_32)
        x_32_sum = x_32_arm + x_gap
        x_32_up = F.interpolate(input=x_32_sum, size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)
        x_16_arm = self.arm16(x_16)
        x_16_sum = x_16_arm + x_32_up
        x_16_up = F.interpolate(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)
        return x_16_up, x_32_up
    '''build the backbone'''
    def buildbackbone(self, cfg):
        from .resnet import BuildResNet
        supported_backbones = {
            'resnet': BuildResNet,
        }
        assert cfg['series'] in supported_backbones, 'unsupport backbone type %s' % cfg['type']
        return supported_backbones[cfg['series']](cfg)


'''Feature Fusion Module to fuse low level output feature of Spatial Path and high level output feature of Context Path'''
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Sigmoid(),
        )
    '''forward'''
    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out


'''BiSeNetV1'''
class BiSeNetV1(nn.Module):
    def __init__(self, backbone_cfg, in_channels=3, spatial_channels_list=(64, 64, 64, 128), context_channels_list=(128, 256, 512),
                 out_indices=(0, 1, 2), out_channels=256, norm_cfg=None, act_cfg=None):
        super(BiSeNetV1, self).__init__()
        assert (len(spatial_channels_list) == 4) and (len(context_channels_list) == 3)
        # set attrs
        self.out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # define modules
        self.context_path = ContextPath(backbone_cfg, context_channels_list, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.spatial_path = SpatialPath(in_channels, spatial_channels_list, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffm = FeatureFusionModule(context_channels_list[1], out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
    '''forward'''
    def forward(self, x):
        x_context8, x_context16 = self.context_path(x)
        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_context8)
        outs = [x_context8, x_context16, x_fuse]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


'''BuildBiSeNetV1'''
def BuildBiSeNetV1(bisenetv1_cfg):
    # assert whether support
    bisenetv1_type = bisenetv1_cfg.pop('type')
    # parse cfg
    default_cfg = {
        'backbone_cfg': None,
        'in_channels': 3, 
        'spatial_channels_list': (64, 64, 64, 128),
        'context_channels_list': (128, 256, 512),
        'out_indices': (0, 1, 2),
        'out_channels': 256,
        'norm_cfg': None, 
        'act_cfg': {'type': 'relu', 'inplace': True},
        'pretrained': False,
        'pretrained_model_path': '',
    }
    for key, value in bisenetv1_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain bisenetv1_cfg
    bisenetv1_cfg = default_cfg.copy()
    pretrained = bisenetv1_cfg.pop('pretrained')
    pretrained_model_path = bisenetv1_cfg.pop('pretrained_model_path')
    # obtain the instanced bisenetv1
    model = BiSeNetV1(**bisenetv1_cfg)
    # load weights of pretrained model
    if pretrained and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        checkpoint = model_zoo.load_url(model_urls[bisenetv1_type])
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model