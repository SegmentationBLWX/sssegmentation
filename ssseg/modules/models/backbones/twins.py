'''
Function:
    Implementation of Twins
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .mit import EfficientMultiheadAttention
from .bricks import PatchEmbed as PatchEmbedBase
from .bricks import BuildNormalization, FFN, BuildDropout, constructnormcfg


'''model urls'''
model_urls = {
    'pcpvt_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/pcpvt_small.pth',
    'pcpvt_base': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/pcpvt_base.pth',
    'pcpvt_large': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/pcpvt_large.pth',
    'svt_small': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/alt_gvt_small.pth',
    'svt_base': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/alt_gvt_base.pth',
    'svt_large': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/alt_gvt_large.pth',
}


'''PatchEmbed'''
class PatchEmbed(PatchEmbedBase):
    def __init__(self, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        if self.norm is None: return {}
        return {'PatchEmbed.norm': self.norm}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'PatchEmbed.projection': self.projection}


'''GlobalSubsampledAttention'''
class GlobalSubsampledAttention(EfficientMultiheadAttention):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_cfg=None, batch_first=True, qkv_bias=True, norm_cfg=None, sr_ratio=1):
        super(GlobalSubsampledAttention, self).__init__(embed_dims, num_heads, attn_drop, proj_drop, dropout_cfg, batch_first, qkv_bias, norm_cfg, sr_ratio)
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        if hasattr(self, 'norm'):
            return {'GlobalSubsampledAttention.norm': self.norm}
        return {}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {'GlobalSubsampledAttention.attn': self.attn}
        if hasattr(self, 'sr'):
            layers.update({'GlobalSubsampledAttention.sr': self.sr})
        return layers


'''GSAEncoderLayer'''
class GSAEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_fcs=2, 
                 qkv_bias=True, act_cfg=None, norm_cfg=None, sr_ratio=1., dropout_cfg=None):
        super(GSAEncoderLayer, self).__init__()
        if dropout_cfg is None: dropout_cfg = {'type': 'droppath', 'drop_prob': drop_path_rate}
        self.norm1 = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
        self.attn = GlobalSubsampledAttention(
            embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate, 
            dropout_cfg=dropout_cfg, qkv_bias=qkv_bias, norm_cfg=norm_cfg, sr_ratio=sr_ratio
        )
        self.norm2 = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
        self.ffn = FFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=num_fcs, ffn_drop=drop_rate,
            dropout_cfg=dropout_cfg, act_cfg=act_cfg, add_identity=False,
        )
        self.drop_path = BuildDropout(dropout_cfg) if (dropout_cfg and (drop_path_rate > 0.)) else nn.Identity()
    '''forward'''
    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape, identity=0.))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        layers = {'GSAEncoderLayer.norm1': self.norm1, 'GSAEncoderLayer.norm2': self.norm2}
        for key, value in self.attn.zerowdlayers().items():
            layers['GSAEncoderLayer.' + key] = value
        return layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {'GSAEncoderLayer.ffn': self.ffn}
        for key, value in self.attn.nonzerowdlayers().items():
            layers['GSAEncoderLayer.' + key] = value
        return layers


'''LocallyGroupedSelfAttention'''
class LocallyGroupedSelfAttention(nn.Module):
    def __init__(self, embed_dims, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_rate=0., proj_drop_rate=0., window_size=1):
        super(LocallyGroupedSelfAttention, self).__init__()
        # set attributes
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # set layers
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
    '''forward'''
    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        x = x.view(b, h, w, c)
        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # calculate attention mask for LSA
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)
        # [B, _h, _w, window_size, window_size, C]
        x = x.reshape(b, _h, self.window_size, _w, self.window_size, c).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w, self.window_size).transpose(2, 3).reshape(1, _h * _w, self.window_size * self.window_size)
        # [1, _h*_w, window_size*window_size, window_size*window_size]
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
        # [3, B, _w*_h, nhead, window_size*window_size, dim]
        qkv = self.qkv(x).reshape(b, _h * _w, self.window_size * self.window_size, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, _h*_w, n_head, window_size*window_size, window_size*window_size]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(b, _h, _w, self.window_size, self.window_size, c)
        x = attn.transpose(2, 3).reshape(b, _h * self.window_size, _w * self.window_size, c)
        if pad_r > 0 or pad_b > 0: x = x[:, :h, :w, :].contiguous()
        x = x.reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        return {}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'LocallyGroupedSelfAttention.qkv': self.qkv, 'LocallyGroupedSelfAttention.proj': self.proj}


'''LSAEncoderLayer'''
class LSAEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 num_fcs=2, qkv_bias=True, qk_scale=None, act_cfg=None, norm_cfg=None, window_size=1, dropout_cfg=None):
        super(LSAEncoderLayer, self).__init__()
        if dropout_cfg is None: dropout_cfg = {'type': 'droppath', 'drop_prob': drop_path_rate}
        self.norm1 = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
        self.attn = LocallyGroupedSelfAttention(embed_dims, num_heads, qkv_bias, qk_scale, attn_drop_rate, drop_rate, window_size)
        self.norm2 = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
        self.ffn = FFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=num_fcs, ffn_drop=drop_rate,
            dropout_cfg=dropout_cfg, act_cfg=act_cfg, add_identity=False,
        )
        self.drop_path = BuildDropout(dropout_cfg) if (dropout_cfg and (drop_path_rate > 0.)) else nn.Identity()
    '''forward'''
    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        layers = {'LSAEncoderLayer.norm1': self.norm1, 'LSAEncoderLayer.norm2': self.norm2}
        for key, value in self.attn.zerowdlayers().items():
            layers['LSAEncoderLayer.' + key] = value
        return layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {'LSAEncoderLayer.ffn': self.ffn}
        for key, value in self.attn.nonzerowdlayers().items():
            layers['LSAEncoderLayer.' + key] = value
        return layers


'''The Conditional Position Encoding (CPE) module'''
class ConditionalPositionEncoding(nn.Module):
    def __init__(self, in_channels, embed_dims=768, stride=1):
        super(ConditionalPositionEncoding, self).__init__()
        self.stride = stride
        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=3, stride=stride, padding=1, bias=True, groups=embed_dims)
    '''forward'''
    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(b, c, h, w)
        if self.stride == 1: x = self.proj(cnn_feat) + cnn_feat
        else: x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        return {}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'ConditionalPositionEncoding.proj': self.proj}


'''Twins-PCPVT'''
class PCPVT(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], patch_sizes=[4, 2, 2, 2], strides=[4, 2, 2, 2], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 out_indices=(0, 1, 2, 3), qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 norm_after_stage=False, norm_cfg=None, act_cfg=None):
        super(PCPVT, self).__init__()
        # set attributes
        self.depths = depths
        self.out_indices = out_indices
        self.norm_after_stage = norm_after_stage
        # patch embed
        self.patch_embeds = nn.ModuleList()
        self.position_encoding_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(
                in_channels=in_channels if i == 0 else embed_dims[i - 1], embed_dims=embed_dims[i],
                kernel_size=patch_sizes[i], stride=strides[i], padding='corner', norm_cfg=norm_cfg
            ))
            self.position_encoding_drops.append(nn.Dropout(p=drop_rate))
        # position encodings
        self.position_encodings = nn.ModuleList([
            ConditionalPositionEncoding(embed_dim, embed_dim) for embed_dim in embed_dims
        ])
        # transformer encoder, stochastic depth decay rule
        self.layers = nn.ModuleList()
        dpr, cur = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))], 0
        for k in range(len(depths)):
            _block = nn.ModuleList([GSAEncoderLayer(
                embed_dims=embed_dims[k], num_heads=num_heads[k], feedforward_channels=mlp_ratios[k] * embed_dims[k],
                attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, drop_path_rate=dpr[cur + i], num_fcs=2, 
                qkv_bias=qkv_bias, act_cfg=act_cfg, norm_cfg=norm_cfg, sr_ratio=sr_ratios[k]) for i in range(depths[k])
            ])
            self.layers.append(_block)
            cur += depths[k]
        # norm
        if self.norm_after_stage:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims: 
                self.norm_list.append(BuildNormalization(constructnormcfg(placeholder=dim, norm_cfg=norm_cfg)))
    '''forward'''
    def forward(self, x):
        outputs, b = list(), x.shape[0]
        for i in range(len(self.depths)):
            x, hw_shape = self.patch_embeds[i](x)
            h, w = hw_shape
            x = self.position_encoding_drops[i](x)
            for j, blk in enumerate(self.layers[i]):
                x = blk(x, hw_shape)
                if j == 0: x = self.position_encodings[i](x, hw_shape)
            if self.norm_after_stage: x = self.norm_list[i](x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outputs.append(x)
        return tuple(outputs)
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        layers = {}
        if hasattr(self, 'norm_list'): layers['PCPVT.norm_list'] = self.norm_list
        for layer_idx, layer in enumerate(self.layers):
            for blk_idx, blk in enumerate(layer):
                for key, value in blk.zerowdlayers().items():
                    assert f'PCPVT.{key}.{layer_idx}.{blk_idx}' not in layers
                    layers[f'PCPVT.{key}.{layer_idx}.{blk_idx}'] = value
        for layer_idx, layer in enumerate(self.position_encodings):
            for key, value in layer.zerowdlayers().items():
                assert f'PCPVT.{key}.{layer_idx}' not in layers
                layers[f'PCPVT.{key}.{layer_idx}'] = value
        for layer_idx, layer in enumerate(self.patch_embeds):
            for key, value in layer.zerowdlayers().items():
                assert f'PCPVT.{key}.{layer_idx}' not in layers
                layers[f'PCPVT.{key}.{layer_idx}'] = value
        return layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {}
        for layer_idx, layer in enumerate(self.layers):
            for blk_idx, blk in enumerate(layer):
                for key, value in blk.nonzerowdlayers().items():
                    assert f'PCPVT.{key}.{layer_idx}.{blk_idx}' not in layers
                    layers[f'PCPVT.{key}.{layer_idx}.{blk_idx}'] = value
        for layer_idx, layer in enumerate(self.position_encodings):
            for key, value in layer.nonzerowdlayers().items():
                assert f'PCPVT.{key}.{layer_idx}' not in layers
                layers[f'PCPVT.{key}.{layer_idx}'] = value
        for layer_idx, layer in enumerate(self.patch_embeds):
            for key, value in layer.nonzerowdlayers().items():
                assert f'PCPVT.{key}.{layer_idx}' not in layers
                layers[f'PCPVT.{key}.{layer_idx}'] = value
        return layers
    '''initialize backbone'''
    def initweights(self, twins_type='pcpvt_small', pretrained_model_path=''):
        # load
        if pretrained_model_path:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(model_urls[twins_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        # be consistent
        state_dict = self.twinsconvert(twins_type, state_dict)
        # load state_dict
        self.load_state_dict(state_dict, strict=False)
    '''twins convert'''
    @staticmethod
    def twinsconvert(twins_type, ckpt):
        new_ckpt = OrderedDict()
        for k, v in list(ckpt.items()):
            new_v = v
            if k.startswith('head'): continue
            elif k.startswith('patch_embeds'):
                if 'proj.' in k: new_k = k.replace('proj.', 'projection.')
                else: new_k = k
            elif k.startswith('blocks'):
                if 'attn.q.' in k:
                    new_k = k.replace('q.', 'attn.in_proj_')
                    new_v = torch.cat([v, ckpt[k.replace('attn.q.', 'attn.kv.')]], dim=0)
                elif 'mlp.fc1' in k:
                    new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in k:
                    new_k = k.replace('mlp.fc2', 'ffn.layers.1')
                elif twins_type.startswith('pcpvt'):
                    if 'attn.proj.' in k: new_k = k.replace('proj.', 'attn.out_proj.')
                    else: new_k = k
                else:
                    if 'attn.proj.' in k:
                        k_lst = k.split('.')
                        if int(k_lst[2]) % 2 == 1: new_k = k.replace('proj.', 'attn.out_proj.')
                        else: new_k = k
                    else:
                        new_k = k
                new_k = new_k.replace('blocks.', 'layers.')
            elif k.startswith('pos_block'):
                new_k = k.replace('pos_block', 'position_encodings')
                if 'proj.0.' in new_k: new_k = new_k.replace('proj.0.', 'proj.')
            else:
                new_k = k
            if 'attn.kv.' not in k: new_ckpt[new_k] = new_v
        return new_ckpt


'''Twins-SVT'''
class SVT(PCPVT):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256], patch_sizes=[4, 2, 2, 2], strides=[4, 2, 2, 2], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                 out_indices=(0, 1, 2, 3), qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, depths=[4, 4, 4], sr_ratios=[4, 2, 1],
                 windiow_sizes=[7, 7, 7], norm_after_stage=True, norm_cfg=None, act_cfg=None):
        super(SVT, self).__init__(
            in_channels=in_channels, embed_dims=embed_dims, patch_sizes=patch_sizes, strides=strides, num_heads=num_heads, mlp_ratios=mlp_ratios,
            out_indices=out_indices, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            depths=depths, sr_ratios=sr_ratios, norm_after_stage=norm_after_stage, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        # transformer encoder, stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for k in range(len(depths)):
            for i in range(depths[k]):
                if i % 2 == 0:
                    self.layers[k][i] = LSAEncoderLayer(
                        embed_dims=embed_dims[k], num_heads=num_heads[k], feedforward_channels=mlp_ratios[k] * embed_dims[k],
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[sum(depths[:k])+i], num_fcs=2,
                        qkv_bias=qkv_bias, window_size=windiow_sizes[k], norm_cfg=norm_cfg, act_cfg=act_cfg
                    )


'''BuildTwins'''
def BuildTwins(twins_cfg):
    # assert whether support
    twins_type = twins_cfg.pop('type')
    supported_twins = {
        'pcpvt_small': [
            PCPVT, {'depths': [3, 4, 6, 3], 'drop_path_rate': 0.2},
        ],
        'pcpvt_base': [
            PCPVT, {'depths': [3, 4, 18, 3], 'drop_path_rate': 0.3},
        ],
        'pcpvt_large': [
            PCPVT, {'depths': [3, 8, 27, 3], 'drop_path_rate': 0.3},
        ],
        'svt_small': [
            SVT, {'embed_dims': [64, 128, 256, 512], 'num_heads': [2, 4, 8, 16], 'mlp_ratios': [4, 4, 4, 4], 'depths': [2, 2, 10, 4], 'windiow_sizes': [7, 7, 7, 7], 'norm_after_stage': True, 'drop_path_rate': 0.2},
        ],
        'svt_base': [
            SVT, {'embed_dims': [96, 192, 384, 768], 'num_heads': [3, 6, 12, 24], 'mlp_ratios': [4, 4, 4, 4], 'depths': [2, 2, 18, 2], 'windiow_sizes': [7, 7, 7, 7], 'norm_after_stage': True, 'drop_path_rate': 0.2},
        ],
        'svt_large': [
            SVT, {'embed_dims': [128, 256, 512, 1024], 'num_heads': [4, 8, 16, 32], 'mlp_ratios': [4, 4, 4, 4], 'depths': [2, 2, 18, 2], 'windiow_sizes': [7, 7, 7, 7], 'norm_after_stage': True, 'drop_path_rate': 0.3},
        ],
    }
    assert twins_type in supported_twins, 'unspport the twins_type %s' % twins_type
    # parse cfg
    default_cfg = {
        'pretrained': True,
        'pretrained_model_path': '',
    }
    if twins_type.startswith('pcpvt'):
        default_cfg.update({
            'in_channels': 3,
            'embed_dims': [64, 128, 320, 512],
            'patch_sizes': [4, 2, 2, 2],
            'strides': [4, 2, 2, 2],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [8, 8, 4, 4],
            'out_indices': (0, 1, 2, 3),
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'sr_ratios': [8, 4, 2, 1],
            'norm_cfg': {'type': 'layernorm'},
            'act_cfg': {'type': 'gelu'},
            'norm_after_stage': False
        })
    else:
        default_cfg.update({
            'in_channels': 3,
            'patch_sizes': [4, 2, 2, 2],
            'strides': [4, 2, 2, 2],
            'out_indices': (0, 1, 2, 3),
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'sr_ratios': [8, 4, 2, 1],
            'norm_cfg': {'type': 'layernorm'},
            'act_cfg': {'type': 'gelu'},
        })
    default_cfg.update(supported_twins[twins_type][1])
    for key, value in twins_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain twins_cfg
    twins_cfg = default_cfg.copy()
    pretrained = twins_cfg.pop('pretrained')
    pretrained_model_path = twins_cfg.pop('pretrained_model_path')
    # obtain the instanced twins
    model = supported_twins[twins_type][0](**twins_cfg)
    # load weights of pretrained model
    if pretrained:
        model.initweights(twins_type, pretrained_model_path)
    # return the model
    return model