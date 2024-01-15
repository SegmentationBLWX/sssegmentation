'''
Function:
    Implementation of BiSeNetV1
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loadpretrainedweights
from .bricks import BuildNormalization, BuildActivation


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''SpatialPath'''
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
                    BuildNormalization(placeholder=num_channels_list[idx], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                )
            elif idx == len(num_channels_list) - 1:
                conv = nn.Sequential(
                    nn.Conv2d(num_channels_list[idx - 1], num_channels_list[idx], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(placeholder=num_channels_list[idx], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(num_channels_list[idx - 1], num_channels_list[idx], kernel_size=3, stride=2, padding=1, bias=False),
                    BuildNormalization(placeholder=num_channels_list[idx], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                )
            self.add_module(layer_name, conv)
    '''forward'''
    def forward(self, x):
        for idx, layer_name in enumerate(self.layers):
            layer_stage = getattr(self, layer_name)
            x = layer_stage(x)
        return x


'''AttentionRefinementModule'''
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(AttentionRefinementModule, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            nn.Sigmoid(),
        )
    '''forward'''
    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


'''ContextPath'''
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
            BuildNormalization(placeholder=context_channels_list[0], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.conv_head16 = nn.Sequential(
            nn.Conv2d(context_channels_list[0], context_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=context_channels_list[0], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(context_channels_list[2], context_channels_list[0], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=context_channels_list[0], norm_cfg=norm_cfg),
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
    '''buildbackbone'''
    def buildbackbone(self, cfg):
        from .resnet import ResNet
        supported_backbones = {
            'ResNet': ResNet,
        }
        backbone_type = cfg.pop('type')
        assert backbone_type, f'unsupport backbone type {backbone_type}'
        return supported_backbones[backbone_type](**cfg)


'''FeatureFusionModule'''
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
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
    def __init__(self, structure_type, backbone_cfg=None, in_channels=3, spatial_channels_list=(64, 64, 64, 128), 
                 context_channels_list=(128, 256, 512), out_indices=(0, 1, 2), out_channels=256, norm_cfg={'type': 'SyncBatchNorm'}, 
                 act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=False, pretrained_model_path=''):
        super(BiSeNetV1, self).__init__()
        assert (len(spatial_channels_list) == 4) and (len(context_channels_list) == 3)
        # set attributes
        self.structure_type = structure_type
        self.backbone_cfg = backbone_cfg
        self.in_channels = in_channels
        self.spatial_channels_list = spatial_channels_list
        self.context_channels_list = context_channels_list
        self.out_indices = out_indices
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # set modules
        self.context_path = ContextPath(backbone_cfg, context_channels_list, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.spatial_path = SpatialPath(in_channels, spatial_channels_list, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffm = FeatureFusionModule(context_channels_list[1], out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        x_context8, x_context16 = self.context_path(x)
        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_context8)
        outs = [x_context8, x_context16, x_fuse]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)