'''
Function:
    Implementation of MIT
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint
from .bricks import BuildNormalization, BuildActivation, BuildDropout, nlctonchw, nchwtonlc, MultiheadAttention, PatchEmbed


'''DEFAULT_MODEL_URLS, the pretrained weights are stored in https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia'''
DEFAULT_MODEL_URLS = {
    'mit-b0': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b0.pth',
    'mit-b1': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b1.pth',
    'mit-b2': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b2.pth',
    'mit-b3': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b3.pth',
    'mit-b4': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b4.pth',
    'mit-b5': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/mit_b5.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {
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
            self.norm = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
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
        self.norm1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate,
            dropout_cfg={'type': 'DropPath', 'drop_prob': drop_path_rate}, batch_first=batch_first,
            qkv_bias=qkv_bias, norm_cfg=norm_cfg, sr_ratio=sr_ratio
        )
        self.norm2 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.ffn = MixFFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels, ffn_drop=drop_rate,
            dropout_cfg={'type': 'DropPath', 'drop_prob': drop_path_rate}, act_cfg=act_cfg
        )
        self.use_checkpoint = use_checkpoint
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
    def __init__(self, structure_type, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], 
                 patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_cfg={'type': 'GELU'}, norm_cfg={'type': 'LayerNorm', 'eps': 1e-6}, 
                 use_checkpoint=False, pretrained=True, pretrained_model_path=''):
        super(MixVisionTransformer, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.out_indices = out_indices
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)
        assert max(out_indices) < num_stages
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur, self.layers = 0, nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels, embed_dims=embed_dims_i, kernel_size=patch_sizes[i], stride=strides[i], padding=patch_sizes[i] // 2, norm_cfg=norm_cfg
            )
            layer = nn.ModuleList([TransformerEncoderLayer(
                embed_dims=embed_dims_i, num_heads=num_heads[i], feedforward_channels=mlp_ratio * embed_dims_i, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[cur + idx], qkv_bias=qkv_bias, act_cfg=act_cfg, norm_cfg=norm_cfg,
                use_checkpoint=use_checkpoint, sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = BuildNormalization(placeholder=embed_dims_i, norm_cfg=norm_cfg)
            self.layers.append(nn.ModuleList([patch_embed, layer, norm]))
            cur += num_layer
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type='', pretrained_model_path=''):
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
        state_dict = self.mitconvert(state_dict)
        self.load_state_dict(state_dict, strict=False)
    '''mitconvert'''
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