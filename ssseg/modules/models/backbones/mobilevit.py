'''
Function:
    Implementation of MobileViT series
Author:
    Zhenchao Jin
'''
import os
import math
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .vit import TransformerEncoderLayer
from .bricks import BuildNormalization, BuildActivation, InvertedResidual


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'mobilevit-small': 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth',
    'mobilevit-xsmall': 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xsmall_3rdparty_in1k_20221018-be39a6e7.pth',
    'mobilevit-xxsmall': 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''MobileVitBlock'''
class MobileVitBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, ffn_dim, out_channels, conv_ksize=3, norm_cfg=dict(type='SyncBatchNorm'), act_cfg=dict(type='Swish'),
                 num_transformer_blocks=2, patch_size=2, num_heads=4, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., no_fusion=False,
                 transformer_norm_cfg=dict(type='LayerNorm')):
        super(MobileVitBlock, self).__init__()
        # build layers
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=conv_ksize, stride=1, padding=int((conv_ksize - 1) / 2), bias=False),
            BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(in_channels, transformer_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        global_rep = [TransformerEncoderLayer(
            embed_dims=transformer_dim, num_heads=num_heads, feedforward_channels=ffn_dim, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, qkv_bias=True, act_cfg=dict(type='Swish'), norm_cfg=transformer_norm_cfg
        ) for _ in range(num_transformer_blocks)]
        global_rep.append(BuildNormalization(placeholder=transformer_dim, norm_cfg=transformer_norm_cfg))
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = nn.Sequential(
            nn.Conv2d(transformer_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=conv_ksize, stride=1, padding=int((conv_ksize - 1) / 2), bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
        # set attributes
        self.patch_size = (patch_size, patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
    '''forward'''
    def forward(self, x):
        shortcut = x
        # local representation
        x = self.local_rep(x)
        # unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        interpolate = False
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True
        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        # global representations
        x = self.global_rep(x)
        # fold (patch -> feature map), [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.conv_proj(x)
        # return
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


'''MobileViT'''
class MobileViT(nn.Module):
    arch_settings = {
        'small': [
            ['mobilenetv2', 32, 1, 1, 4], ['mobilenetv2', 64, 2, 3, 4], ['mobilevit', 96, 2, 144, 288, 2, 4],
            ['mobilevit', 128, 2, 192, 384, 4, 4], ['mobilevit', 160, 2, 240, 480, 3, 4],
        ],
        'x_small': [
            ['mobilenetv2', 32, 1, 1, 4], ['mobilenetv2', 48, 2, 3, 4], ['mobilevit', 64, 2, 96, 192, 2, 4],
            ['mobilevit', 80, 2, 120, 240, 4, 4], ['mobilevit', 96, 2, 144, 288, 3, 4],
        ],
        'xx_small': [
            ['mobilenetv2', 16, 1, 1, 2], ['mobilenetv2', 24, 2, 3, 2], ['mobilevit', 48, 2, 64, 128, 2, 2],
            ['mobilevit', 64, 2, 80, 160, 4, 2], ['mobilevit', 80, 2, 96, 192, 3, 2],
        ],
    }
    def __init__(self, arch='small', in_channels=3, stem_channels=16, last_exp_factor=4, out_indices=(4, ), norm_cfg=dict(type='SyncBatchNorm'), act_cfg=dict(type='Swish')):
        super(MobileViT, self).__init__()
        # assert
        arch = arch.lower()
        assert arch in self.arch_settings
        arch = self.arch_settings[arch]
        if isinstance(out_indices, int): out_indices = [out_indices]
        assert isinstance(out_indices, collections.Sequence)
        # set attributes
        self.arch = arch
        self.num_stages = len(arch)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        # build layers
        _make_layer_func = {
            'mobilenetv2': self.makemobilenetv2layer, 'mobilevit': self.makemobilevitlayer,
        }
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels, *settings)
            layers.append(layer)
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.conv_1x1_exp = nn.Sequential(
            nn.Conv2d(in_channels, last_exp_factor * in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=last_exp_factor * in_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''makemobilevitlayer'''
    @staticmethod
    def makemobilevitlayer(in_channels, out_channels, stride, transformer_dim, ffn_dim, num_transformer_blocks, expand_ratio=4):
        layer = []
        layer.append(InvertedResidual(
            in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=dict(type='Swish'),
        ))
        layer.append(MobileVitBlock(
            in_channels=out_channels, transformer_dim=transformer_dim, ffn_dim=ffn_dim, out_channels=out_channels, num_transformer_blocks=num_transformer_blocks,
        ))
        return nn.Sequential(*layer), out_channels
    '''makemobilenetv2layer'''
    @staticmethod
    def makemobilenetv2layer(in_channels, out_channels, stride, num_blocks, expand_ratio=4):
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layer.append(InvertedResidual(
                in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=dict(type='Swish'),
            ))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels
    '''forward'''
    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.conv_1x1_exp(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


'''MobileViTV2'''
class MobileViTV2(nn.Module):
    def __init__(self):
        super(MobileViTV2, self).__init__()