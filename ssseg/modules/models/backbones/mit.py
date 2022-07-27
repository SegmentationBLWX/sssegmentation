'''
Function:
    Implementation of MIT
Author:
    Zhenchao Jin
'''
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint
from .bricks import PatchEmbed as PatchEmbedBase
from .bricks import BuildNormalization, BuildActivation, BuildDropout, nlctonchw, nchwtonlc, MultiheadAttention, constructnormcfg


'''model urls, the pretrained weights are stored in https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia'''
model_urls = {
    'mit-b0': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b0.pth',
    'mit-b1': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b1.pth',
    'mit-b2': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b2.pth',
    'mit-b3': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b3.pth',
    'mit-b4': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b4.pth',
    'mit-b5': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b5.pth',
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


'''MixFFN'''
class MixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, act_cfg=None, ffn_drop=0., dropout_cfg=None):
        super(MixFFN, self).__init__()
        # set attrs
        self.act_cfg = act_cfg
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        # define layers
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(feedforward_channels, feedforward_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=feedforward_channels),
            BuildActivation(act_cfg),
            nn.Dropout(ffn_drop),
            nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout(ffn_drop),
        )
        # define dropout layer
        self.dropout_layer = BuildDropout(dropout_cfg) if dropout_cfg else torch.nn.Identity()
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        return {}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'MixFFN.layers': self.layers}
    '''forward'''
    def forward(self, x, hw_shape, identity=None):
        out = nlctonchw(x, hw_shape)
        out = self.layers(out)
        out = nchwtonlc(out)
        if identity is None: identity = x
        return identity + self.dropout_layer(out)


'''EfficientMultiheadAttention'''
class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_cfg=None, batch_first=True, qkv_bias=False, norm_cfg=None, sr_ratio=1):
        super(EfficientMultiheadAttention, self).__init__(embed_dims, num_heads, attn_drop, proj_drop, dropout_cfg=dropout_cfg, batch_first=batch_first, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio, padding=0)
            self.norm = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        if hasattr(self, 'norm'):
            return {'EfficientMultiheadAttention.norm': self.norm}
        return {}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {'EfficientMultiheadAttention.attn': self.attn}
        if hasattr(self, 'sr'):
            layers.update({'EfficientMultiheadAttention.sr': self.sr})
        return layers
    '''forward'''
    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlctonchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchwtonlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None: identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first: out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


'''TransformerEncoderLayer'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 qkv_bias=True, act_cfg=None, norm_cfg=None, batch_first=True, sr_ratio=1, use_checkpoint=False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_cfg={'type': 'droppath', 'drop_prob': drop_path_rate},
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio
        )
        self.norm2 = BuildNormalization(constructnormcfg(placeholder=embed_dims, norm_cfg=norm_cfg))
        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_cfg={'type': 'droppath', 'drop_prob': drop_path_rate},
            act_cfg=act_cfg
        )
        self.use_checkpoint = use_checkpoint
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        layers = {
            'TransformerEncoderLayer.norm1': self.norm1,
            'TransformerEncoderLayer.norm2': self.norm2,
        }
        for key, value in self.attn.zerowdlayers().items():
            layers['TransformerEncoderLayer.' + key] = value
        for key, value in self.ffn.zerowdlayers().items():
            layers['TransformerEncoderLayer.' + key] = value
        return layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {}
        for key, value in self.attn.nonzerowdlayers().items():
            layers['TransformerEncoderLayer.' + key] = value
        for key, value in self.ffn.nonzerowdlayers().items():
            layers['TransformerEncoderLayer.' + key] = value
        return layers
    '''forward'''
    def forward(self, x, hw_shape):
        def _forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(_forward, x)
        else:
            x = _forward(x)
        return x


'''MixVisionTransformer'''
class MixVisionTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 act_cfg=None, norm_cfg=None, use_checkpoint=False):
        super(MixVisionTransformer, self).__init__()
        # assert
        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)
        assert max(out_indices) < num_stages
        # set attrs
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur, self.layers = 0, nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg
            )
            layer = nn.ModuleList([TransformerEncoderLayer(
                embed_dims=embed_dims_i,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * embed_dims_i,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[cur + idx],
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                use_checkpoint=use_checkpoint,
                sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = BuildNormalization(constructnormcfg(placeholder=embed_dims_i, norm_cfg=norm_cfg))
            self.layers.append(nn.ModuleList([patch_embed, layer, norm]))
            cur += num_layer
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        zwd_layers = {}
        for layer_idx, layer in enumerate(self.layers):
            assert len(layer) == 3
            for key, value in layer[0].zerowdlayers().items():
                zwd_layers[f'MixVisionTransformer.{layer_idx}_{key}'] = value
            for trans_idx, trans in enumerate(layer[1]):
                for key, value in trans.zerowdlayers().items():
                    zwd_layers[f'MixVisionTransformer.{layer_idx}_{trans_idx}_{key}'] = value
            zwd_layers[f'MixVisionTransformer.{layer_idx}_norm'] = layer[2]
        return zwd_layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        nonzwd_layers = {}
        for layer_idx, layer in enumerate(self.layers):
            assert len(layer) == 3
            for key, value in layer[0].nonzerowdlayers().items():
                nonzwd_layers[f'MixVisionTransformer.{layer_idx}_{key}'] = value
            for trans_idx, trans in enumerate(layer[1]):
                for key, value in trans.nonzerowdlayers().items():
                    nonzwd_layers[f'MixVisionTransformer.{layer_idx}_{trans_idx}_{key}'] = value
        return nonzwd_layers
    '''init weights'''
    def initweights(self, mit_type='', pretrained_model_path=''):
        if pretrained_model_path:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(model_urls[mit_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.mitconvert(state_dict)
        self.load_state_dict(state_dict, strict=False)
    '''mit convert'''
    @staticmethod
    def mitconvert(ckpt):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        # process the concat between q linear weights and kv linear weights
        for k, v in ckpt.items():
            if k.startswith('head'): continue
            # patch embedding convertion
            elif k.startswith('patch_embed'):
                stage_i = int(k.split('.')[0].replace('patch_embed', ''))
                new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i-1}.0')
                new_v = v
                if 'proj.' in new_k: new_k = new_k.replace('proj.', 'projection.')
            # transformer encoder layer convertion
            elif k.startswith('block'):
                stage_i = int(k.split('.')[0].replace('block', ''))
                new_k = k.replace(f'block{stage_i}', f'layers.{stage_i-1}.1')
                new_v = v
                if 'attn.q.' in new_k:
                    sub_item_k = k.replace('q.', 'kv.')
                    new_k = new_k.replace('q.', 'attn.in_proj_')
                    new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
                elif 'attn.kv.' in new_k: continue
                elif 'attn.proj.' in new_k: new_k = new_k.replace('proj.', 'attn.out_proj.')
                elif 'attn.sr.' in new_k: new_k = new_k.replace('sr.', 'sr.')
                elif 'mlp.' in new_k:
                    string = f'{new_k}-'
                    new_k = new_k.replace('mlp.', 'ffn.layers.')
                    if 'fc1.weight' in new_k or 'fc2.weight' in new_k: new_v = v.reshape((*v.shape, 1, 1))
                    new_k = new_k.replace('fc1.', '0.')
                    new_k = new_k.replace('dwconv.dwconv.', '1.')
                    new_k = new_k.replace('fc2.', '4.')
                    string += f'{new_k} {v.shape}-{new_v.shape}'
            # norm layer convertion
            elif k.startswith('norm'):
                stage_i = int(k.split('.')[0].replace('norm', ''))
                new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i-1}.2')
                new_v = v
            else:
                new_k = k
                new_v = v
            new_ckpt[new_k] = new_v
        return new_ckpt
    '''forward'''
    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]: x = block(x, hw_shape)
            x = layer[2](x)
            x = nlctonchw(x, hw_shape)
            if i in self.out_indices: outs.append(x)
        return outs


'''BuildMixVisionTransformer'''
def BuildMixVisionTransformer(mit_cfg):
    # assert whether support
    mit_type = mit_cfg.pop('type')
    supported_mits = {
        'mit-b0': {
            'embed_dims': 32, 'num_stages': 4, 'num_layers': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
        },
        'mit-b1': {
            'embed_dims': 64, 'num_stages': 4, 'num_layers': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
        },
        'mit-b2': {
            'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 4, 6, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
        },
        'mit-b3': {
            'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 4, 18, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
        },
        'mit-b4': {
            'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 8, 27, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
        },
        'mit-b5': {
            'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 6, 40, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
        },
    }
    assert mit_type in supported_mits, 'unspport the mit_type %s' % mit_type
    # parse cfg
    default_cfg = {
        'in_channels': 3,
        'strides': [4, 2, 2, 2],
        'out_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
        'act_cfg': {'type': 'gelu'},
        'pretrained': True,
        'pretrained_model_path': '',
        'use_checkpoint': False,
    }
    default_cfg.update(supported_mits[mit_type])
    for key, value in mit_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain mit_cfg
    mit_cfg = default_cfg.copy()
    pretrained = mit_cfg.pop('pretrained')
    pretrained_model_path = mit_cfg.pop('pretrained_model_path')
    # obtain the instanced mit
    model = MixVisionTransformer(**mit_cfg)
    # load weights of pretrained model
    if pretrained:
        model.initweights(mit_type, pretrained_model_path)
    # return the model
    return model