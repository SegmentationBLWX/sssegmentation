'''
Function:
    Implementation of ConvNeXt
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from functools import partial
from .bricks.dropout.droppath import DropPath
from .bricks import BuildNormalization, BuildActivation, constructnormcfg


'''model urls'''
model_urls = {
    'convnext_tiny': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
    'convnext_small': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth',
    'convnext_base': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
    'convnext_base_21k': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
    'convnext_large_21k': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth',
    'convnext_xlarge_21k': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth',
}


'''LayerNorm2d'''
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, **kwargs):
        super(LayerNorm2d, self).__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]
    '''forward'''
    def forward(self, x):
        assert x.dim() == 4, f'LayerNorm2d only supports inputs with shape (N, C, H, W), but got tensor with shape {x.shape}'
        x = F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)
        return x


'''ConvNeXtBlock'''
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, norm_cfg=None, act_cfg=None, mlp_ratio=4., linear_pw_conv=True, drop_path_rate=0., layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.linear_pw_conv = linear_pw_conv
        if norm_cfg['type'] in ['layernorm2d']:
            norm_cfg_copy = copy.deepcopy(norm_cfg)
            norm_cfg_copy.pop('type')
            self.norm = LayerNorm2d(num_channels=in_channels, **norm_cfg_copy)
        else:
            self.norm = BuildNormalization(constructnormcfg(placeholder=in_channels, norm_cfg=norm_cfg))
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
        'tiny': {
            'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3], 'channels': [256, 512, 1024, 2048]
        },
    }
    def __init__(self, arch='tiny', in_channels=3, stem_patch_size=4, norm_cfg=None, act_cfg=None, linear_pw_conv=True,
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=-1, gap_before_final_norm=True):
        super(ConvNeXt, self).__init__()
        arch = self.arch_settings[arch]
        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.gap_before_final_norm = gap_before_final_norm
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = nn.ModuleList()
        if norm_cfg['type'] in ['layernorm2d']:
            norm_cfg_copy = copy.deepcopy(norm_cfg)
            norm_cfg_copy.pop('type')
            norm_layer = LayerNorm2d(num_channels=self.channels[0], **norm_cfg_copy)
        else:
            norm_layer = BuildNormalization(constructnormcfg(placeholder=self.channels[0], norm_cfg=norm_cfg))
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
                if norm_cfg['type'] in ['layernorm2d']:
                    norm_cfg_copy = copy.deepcopy(norm_cfg)
                    norm_cfg_copy.pop('type')
                    norm_layer = LayerNorm2d(num_channels=channels, **norm_cfg_copy)
                else:
                    norm_layer = BuildNormalization(constructnormcfg(placeholder=channels, norm_cfg=norm_cfg))
                self.add_module(f'norm{i}', norm_layer)
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
    '''initweights'''
    def initweights(self, convnext_type, pretrained_model_path=''):
        if pretrained_model_path:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(model_urls[convnext_type], map_location='cpu')
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


'''BuildConvNeXt'''
def BuildConvNeXt(convnext_cfg):
    # assert whether support
    convnext_type = convnext_cfg.pop('type')
    # parse cfg
    default_cfg = {
        'arch': 'tiny', 
        'in_channels': 3, 
        'stem_patch_size': 4, 
        'norm_cfg': {'type': 'layernorm2d', 'eps': 1e-6},
        'act_cfg': {'type': 'gelu'},
        'linear_pw_conv': True,
        'drop_path_rate': 0., 
        'layer_scale_init_value': 1e-6, 
        'out_indices': (0, 1, 2, 3), 
        'gap_before_final_norm': True,
        'pretrained': True,
        'pretrained_model_path': '',
    }
    for key, value in convnext_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain convnext_cfg
    convnext_cfg = default_cfg.copy()
    pretrained = convnext_cfg.pop('pretrained')
    pretrained_model_path = convnext_cfg.pop('pretrained_model_path')
    # obtain the instanced convnext
    model = ConvNeXt(**convnext_cfg)
    # load weights of pretrained model
    if pretrained:
        model.initweights(convnext_type, pretrained_model_path)
    # return the model
    return model