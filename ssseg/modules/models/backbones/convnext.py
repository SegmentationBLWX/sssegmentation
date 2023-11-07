'''
Function:
    Implementation of ConvNeXt
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from functools import partial
from .bricks.dropout.droppath import DropPath
from .bricks import BuildNormalization, BuildActivation
from .bricks.normalization.layernorm2d import LayerNorm2d


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'convnext_tiny': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
    'convnext_small': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth',
    'convnext_base': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
    'convnext_base_21k': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
    'convnext_large_21k': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth',
    'convnext_xlarge_21k': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''ConvNeXtBlock'''
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, norm_cfg=None, act_cfg=None, mlp_ratio=4., linear_pw_conv=True, drop_path_rate=0., layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.linear_pw_conv = linear_pw_conv
        self.norm = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg)
        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = BuildActivation(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    '''forward'''
    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))
        x = shortcut + self.drop_path(x)
        return x


'''ConvNeXt'''
class ConvNeXt(nn.Module):
    arch_settings = {
        'tiny': {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]},
        'xlarge': {'depths': [3, 3, 27, 3], 'channels': [256, 512, 1024, 2048]},
    }
    def __init__(self, structure_type, arch='tiny', in_channels=3, stem_patch_size=4, norm_cfg={'type': 'LayerNorm2d', 'eps': 1e-6}, act_cfg={'type': 'GELU'},
                 linear_pw_conv=True, drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=(0, 1, 2, 3), gap_before_final_norm=True,
                 pretrained=True, pretrained_model_path=''):
        super(ConvNeXt, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.arch = arch
        self.in_channels = in_channels
        self.stem_patch_size = stem_patch_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.linear_pw_conv = linear_pw_conv
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.gap_before_final_norm = gap_before_final_norm
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        arch = self.arch_settings[arch]
        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = nn.ModuleList()
        norm_layer = BuildNormalization(placeholder=self.channels[0], norm_cfg=norm_cfg)
        stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=stem_patch_size, stride=stem_patch_size),
            norm_layer,
        )
        self.downsample_layers.append(stem)
        # 4 feature resolution stages, each consisting of multiple residual blocks
        block_idx = 0
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(self.channels[i - 1], channels, kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels, drop_path_rate=dpr[block_idx + j], norm_cfg=norm_cfg, act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv, layer_scale_init_value=layer_scale_init_value
                ) for j in range(depth)
            ])
            block_idx += depth
            self.stages.append(stage)
            if i in self.out_indices:
                norm_layer = BuildNormalization(placeholder=channels, norm_cfg=norm_cfg)
                self.add_module(f'norm{i}', norm_layer)
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''forward'''
    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type, pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict_convert = {}
        for key, value in state_dict.items():
            state_dict_convert[key.replace('backbone.', '')] = value
        self.load_state_dict(state_dict_convert, strict=False)