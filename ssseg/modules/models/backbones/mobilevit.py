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
from .bricks import BuildNormalization, BuildActivation, InvertedResidual, BuildDropout, makedivisible


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'mobilevit-small': 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth',
    'mobilevit-xsmall': 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xsmall_3rdparty_in1k_20221018-be39a6e7.pth',
    'mobilevit-xxsmall': 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth',
    'mobilevitv2_050': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_050-49951ee2.pth',
    'mobilevitv2_075': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_075-b5556ef6.pth',
    'mobilevitv2_100': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_100-e464ef3b.pth',
    'mobilevitv2_125': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_125-0ae35027.pth',
    'mobilevitv2_150': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_150-737c5019.pth',
    'mobilevitv2_175': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_175-16462ee2.pth',
    'mobilevitv2_200': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_200-b3422f67.pth',
    'mobilevitv2_150_in22ft1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_150_in22ft1k-0b555d7b.pth',
    'mobilevitv2_175_in22ft1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_175_in22ft1k-4117fa1f.pth',
    'mobilevitv2_200_in22ft1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_200_in22ft1k-1d7c8927.pth',
    'mobilevitv2_150_384_in22ft1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_150_384_in22ft1k-9e142854.pth',
    'mobilevitv2_175_384_in22ft1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_175_384_in22ft1k-059cbe56.pth',
    'mobilevitv2_200_384_in22ft1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_200_384_in22ft1k-32c87503.pth',
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
            nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=conv_ksize, stride=1, padding=int((conv_ksize - 1) / 2), bias=False)),
                ('bn', BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg)),
                ('activate', BuildActivation(act_cfg)),
            ])),
            nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels, transformer_dim, kernel_size=1, stride=1, padding=0, bias=False))
            ])),
        )
        global_rep = [TransformerEncoderLayer(
            embed_dims=transformer_dim, num_heads=num_heads, feedforward_channels=ffn_dim, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, qkv_bias=True, act_cfg=dict(type='Swish'), norm_cfg=transformer_norm_cfg
        ) for _ in range(num_transformer_blocks)]
        global_rep.append(BuildNormalization(placeholder=transformer_dim, norm_cfg=transformer_norm_cfg))
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(transformer_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)),
            
        ]))
        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=conv_ksize, stride=1, padding=int((conv_ksize - 1) / 2), bias=False)),
                ('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)),
                ('activate', BuildActivation(act_cfg)),
            ]))
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
    def __init__(self, structure_type, arch='small', in_channels=3, stem_channels=16, last_exp_factor=4, out_indices=(0, 1, 2, 3, 4), norm_cfg=dict(type='SyncBatchNorm'), 
                 act_cfg=dict(type='Swish'), pretrained=True, pretrained_model_path=''):
        super(MobileViT, self).__init__()
        # assert
        arch = arch.lower()
        assert arch in self.arch_settings
        arch = self.arch_settings[arch]
        if isinstance(out_indices, int): out_indices = [out_indices]
        assert isinstance(out_indices, collections.abc.Sequence)
        # set attributes
        self.arch = arch
        self.num_stages = len(arch)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # build layers
        _make_layer_func = {
            'mobilenetv2': self.makemobilenetv2layer, 'mobilevit': self.makemobilevitlayer,
        }
        self.stem = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn', BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg)),
            ('activate', BuildActivation(act_cfg)),
        ]))
        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels, norm_cfg, act_cfg, *settings)
            layers.append(layer)
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.conv_1x1_exp = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, last_exp_factor * in_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn', BuildNormalization(placeholder=last_exp_factor * in_channels, norm_cfg=norm_cfg)),
            ('activate', BuildActivation(act_cfg)),
        ]))
        # load pretrained weights
        if pretrained and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = self.convertstatedict(checkpoint)
            self.load_state_dict(state_dict, strict=True)
        elif pretrained:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
            state_dict = self.convertstatedict(checkpoint)
            self.load_state_dict(state_dict, strict=True)
    '''convertstatedict'''
    @staticmethod
    def convertstatedict(checkpoint):
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
            if key.startswith('head'):
                state_dict.pop(key)
            if 'attn.qkv' in key:
                new_key = key.replace('attn.qkv.', 'attn.attn.in_proj_')
                assert new_key not in state_dict
                state_dict[new_key] = state_dict.pop(key)
            if 'attn.proj' in key:
                new_key = key.replace('attn.proj', 'attn.attn.out_proj')
                assert new_key not in state_dict
                state_dict[new_key] = state_dict.pop(key)
        return state_dict
    '''makemobilevitlayer'''
    @staticmethod
    def makemobilevitlayer(in_channels, norm_cfg, act_cfg, out_channels, stride, transformer_dim, ffn_dim, num_transformer_blocks, expand_ratio=4):
        layer = []
        layer.append(InvertedResidual(
            in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg,
        ))
        layer.append(MobileVitBlock(
            in_channels=out_channels, transformer_dim=transformer_dim, ffn_dim=ffn_dim, out_channels=out_channels, num_transformer_blocks=num_transformer_blocks,
            act_cfg=act_cfg, norm_cfg=norm_cfg, 
        ))
        return nn.Sequential(*layer), out_channels
    '''makemobilenetv2layer'''
    @staticmethod
    def makemobilenetv2layer(in_channels, norm_cfg, act_cfg, out_channels, stride, num_blocks, expand_ratio=4):
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layer.append(InvertedResidual(
                in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg,
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


'''LinearSelfAttention'''
class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_drop=0.0, proj_drop=0.0, bias=True):
        super(LinearSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim, out_channels=1 + (2 * embed_dim), bias=bias, kernel_size=1,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim, out_channels=embed_dim, bias=bias, kernel_size=1,
        )
        self.out_drop = nn.Dropout(proj_drop)
    '''forwardselfattn'''
    def forwardselfattn(self, x):
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)
        # Project x into query, key and value, Query --> [B, 1, P, N], value, key --> [B, d, P, N]
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)
        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        # Compute context vector, [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        # combine context vector with values, [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        # return
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out
    '''forwardcrossattn'''
    @torch.jit.ignore()
    def forwardcrossattn(self, x, x_prev=None):
        # x --> [B, C, P, N], x_prev = [B, C, P, M]
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]
        assert (kv_patch_area == q_patch_area), 'the number of pixels in a patch for query and key_value should be the same'
        # compute query, key, and value, [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(x_prev, weight=self.qkv_proj.weight[:self.embed_dim + 1], bias=self.qkv_proj.bias[:self.embed_dim + 1])
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = qk.split([1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(x, weight=self.qkv_proj.weight[self.embed_dim + 1], bias=self.qkv_proj.bias[self.embed_dim + 1] if self.qkv_proj.bias is not None else None)
        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        # compute context vector, [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        # combine context vector with values, [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        # return
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out
    '''forward'''
    def forward(self, x, x_prev=None):
        if x_prev is None:
            return self.forwardselfattn(x)
        else:
            return self.forwardcrossattn(x, x_prev=x_prev)


'''ConvMlp'''
class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg={'type': 'ReLU', 'inplace': True}, norm_cfg=None, bias=True, drop=0.):
        super(ConvMlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = self.totuple(bias)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = BuildNormalization(placeholder=hidden_features, norm_cfg=norm_cfg)
        self.act = BuildActivation(act_cfg)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])
    '''forward'''
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    '''totuple'''
    @staticmethod
    def totuple(x):
        if isinstance(x, (int, bool, float)): return (x, x)
        assert isinstance(x, tuple) and (len(x) == 2)
        return x


'''LinearTransformerBlock'''
class LinearTransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=2.0, drop=0.0, attn_drop=0.0, drop_path=0.0, act_cfg={'type': 'SiLU'}, norm_cfg={'type': 'GroupNorm', 'num_groups': 1}):
        super(LinearTransformerBlock, self).__init__()
        self.norm1 = BuildNormalization(placeholder=embed_dim, norm_cfg=norm_cfg)
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = BuildDropout({'type': 'DropPath', 'drop_prob': drop_path})
        self.norm2 = BuildNormalization(placeholder=embed_dim, norm_cfg=norm_cfg)
        self.mlp = ConvMlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_cfg=act_cfg, drop=drop)
        self.drop_path2 = BuildDropout({'type': 'DropPath', 'drop_prob': drop_path})
    '''forward'''
    def forward(self, x, x_prev=None):
        # self-attention
        if x_prev is None:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        # cross-attention
        else:
            res = x
            x = self.norm1(x)
            x = self.attn(x, x_prev)
            x = self.drop_path1(x) + res
        # feed forward network
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        # return
        return x


'''MobileVitV2Block'''
class MobileVitV2Block(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bottle_ratio=1.0, group_size=1, dilation=1, mlp_ratio=2.0, transformer_dim=None,
                 transformer_depth=2, patch_size=2, attn_drop=0., drop=0., drop_path_rate=0., transformer_norm_cfg={'type': 'GroupNorm', 'num_groups': 1},
                 norm_cfg=dict(type='SyncBatchNorm'), act_cfg={'type': 'ReLU', 'inplace': True}):
        super(MobileVitV2Block, self).__init__()
        if not group_size:
            groups = 1
        else:
            groups = in_channels // group_size
        out_channels = out_channels or in_channels
        transformer_dim = transformer_dim or makedivisible(bottle_ratio * in_channels, 8)
        # build layers
        assert int((kernel_size - 1) / 2) == dilation
        self.conv_kxk = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=groups, padding=int((kernel_size - 1) / 2), bias=False, dilation=dilation)),
            ('bn', BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg)),
            ('activate', BuildActivation(act_cfg)),
        ]))
        self.conv_1x1 = nn.Conv2d(in_channels, transformer_dim, kernel_size=1, bias=False)
        self.transformer = nn.Sequential(*[LinearTransformerBlock(
            embed_dim=transformer_dim, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop,
            drop_path=drop_path_rate, act_cfg=act_cfg, norm_cfg=transformer_norm_cfg
        ) for _ in range(transformer_depth)])
        self.norm = BuildNormalization(placeholder=transformer_dim, norm_cfg=transformer_norm_cfg)
        self.conv_proj = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(transformer_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)),
        ]))
        # set attributes
        self.patch_size = ConvMlp.totuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
    '''forward'''
    def forward(self, x):
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        # local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)
        # unfold (feature map -> patches), [B, C, H, W] -> [B, C, P, N]
        C = x.shape[1]
        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)
        # global representations
        x = self.transformer(x)
        x = self.norm(x)
        # fold (patches -> feature map), [B, C, P, N] --> [B, C, H, W]
        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w).permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        # return
        x = self.conv_proj(x)
        return x


'''MobileViTV2'''
class MobileViTV2(nn.Module):
    arch_settings = {
        'mobilevitv2_050': [[
            ['mobilenetv2', 32, 1, 1, 2], ['mobilenetv2', 64, 2, 2, 2], ['mobilevitv2', 128, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 192, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 256, 2, 0.5, 3, 1, 2, 2],
        ], 16],
        'mobilevitv2_075': [[
            ['mobilenetv2', 48, 1, 1, 2], ['mobilenetv2', 96, 2, 2, 2], ['mobilevitv2', 192, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 288, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 384, 2, 0.5, 3, 1, 2, 2],
        ], 24],
        'mobilevitv2_100': [[
            ['mobilenetv2', 64, 1, 1, 2], ['mobilenetv2', 128, 2, 2, 2], ['mobilevitv2', 256, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 384, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 512, 2, 0.5, 3, 1, 2, 2],
        ], 32],
        'mobilevitv2_125': [[
            ['mobilenetv2', 80, 1, 1, 2], ['mobilenetv2', 160, 2, 2, 2], ['mobilevitv2', 320, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 480, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 640, 2, 0.5, 3, 1, 2, 2],
        ], 40],
        'mobilevitv2_150': [[
            ['mobilenetv2', 96, 1, 1, 2], ['mobilenetv2', 192, 2, 2, 2], ['mobilevitv2', 384, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 576, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 768, 2, 0.5, 3, 1, 2, 2],
        ], 48],
        'mobilevitv2_175': [[
            ['mobilenetv2', 112, 1, 1, 2], ['mobilenetv2', 224, 2, 2, 2], ['mobilevitv2', 448, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 672, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 896, 2, 0.5, 3, 1, 2, 2],
        ], 56],
        'mobilevitv2_200': [[
            ['mobilenetv2', 128, 1, 1, 2], ['mobilenetv2', 256, 2, 2, 2], ['mobilevitv2', 512, 2, 0.5, 2, 1, 2, 2],
            ['mobilevitv2', 768, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 1024, 2, 0.5, 3, 1, 2, 2],
        ], 64],
    }
    def __init__(self, structure_type, arch='mobilevitv2_050', in_channels=3, out_indices=(0, 1, 2, 3, 4), norm_cfg=dict(type='SyncBatchNorm'), 
                 act_cfg=dict(type='SiLU', inplace=True), pretrained=True, pretrained_model_path=''):
        super(MobileViTV2, self).__init__()
        # assert
        arch = arch.lower()
        assert arch in self.arch_settings
        arch, stem_channels = self.arch_settings[arch]
        if isinstance(out_indices, int): out_indices = [out_indices]
        assert isinstance(out_indices, collections.abc.Sequence)
        # set attributes
        self.arch = arch
        self.num_stages = len(arch)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # build stem
        self.stem = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn', BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg)),
            ('activate', BuildActivation(act_cfg)),
        ]))
        # build stages
        _make_layer_func = {
            'mobilenetv2': self.makemobilenetv2layer, 'mobilevitv2': self.makemobilevitv2layer,
        }
        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels, norm_cfg, act_cfg, *settings)
            layers.append(layer)
            in_channels = out_channels
        self.stages = nn.Sequential(*layers)
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type='mobilevit-small', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        state_dict = self.convertstatedict(checkpoint)
        self.load_state_dict(state_dict, strict=True)
    '''convertstatedict'''
    @staticmethod
    def convertstatedict(checkpoint):
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('head'):
                state_dict.pop(key)
            if 'conv1_1x1' in key:
                new_key = key.replace('conv1_1x1', 'conv.0')
                assert new_key not in state_dict, new_key
                state_dict[new_key] = state_dict.pop(key)
            if 'conv2_kxk' in key:
                new_key = key.replace('conv2_kxk', 'conv.1')
                assert new_key not in state_dict, new_key
                state_dict[new_key] = state_dict.pop(key)
            if 'conv3_1x1' in key:
                new_key = key.replace('conv3_1x1', 'conv.2')
                assert new_key not in state_dict, new_key
                state_dict[new_key] = state_dict.pop(key)
        return state_dict
    '''makemobilenetv2layer'''
    @staticmethod
    def makemobilenetv2layer(in_channels, norm_cfg, act_cfg, out_channels, stride, num_blocks, expand_ratio=2):
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layer.append(InvertedResidual(
                in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg,
            ))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels
    '''makemobilevitv2layer'''
    @staticmethod
    def makemobilevitv2layer(in_channels, norm_cfg, act_cfg, out_channels, stride, bottle_ratio, transformer_depth, num_transformer_blocks, mlp_ratio, expand_ratio=2):
        layer = []
        layer.append(InvertedResidual(
            in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg,
        ))
        for i in range(num_transformer_blocks):
            layer.append(MobileVitV2Block(
                in_channels=out_channels, out_channels=out_channels, transformer_depth=transformer_depth, 
                bottle_ratio=bottle_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg, mlp_ratio=mlp_ratio,
            ))
        return nn.Sequential(*layer), out_channels
    '''forward'''
    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)