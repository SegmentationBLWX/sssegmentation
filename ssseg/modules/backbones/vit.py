'''
Function:
    Implementation of ViT
Author:
    Zhenchao Jin
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint
from .bricks import BuildNormalization, BuildActivation, MultiheadAttention, FFN, PatchEmbed


'''model urls'''
model_urls = {
    'jx_vit_large_p16_384': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
}


'''Implements one encoder layer in Vision Transformer'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_fcs=2, qkv_bias=True, act_cfg=None, norm_cfg=None, batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.ln1 = BuildNormalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_cfg={'type': 'droppath', 'opts': {'drop_prob': drop_path_rate}},
            batch_first=batch_first,
            bias=qkv_bias
        )
        self.ln2 = BuildNormalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_cfg={'type': 'droppath', 'opts': {'drop_prob': drop_path_rate}},
            act_cfg=act_cfg
        )
    '''forward'''
    def forward(self, x):
        x = self.attn(self.ln1(x), identity=x)
        x = self.ffn(self.ln2(x), identity=x)
        return x


'''Vision Transformer'''
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dims=768, num_layers=12, num_heads=12, mlp_ratio=4, out_indices=-1, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., with_cls_token=True, output_cls_token=False, norm_cfg=None, act_cfg=None, 
                 patch_norm=False, final_norm=False, interpolate_mode='bicubic', num_fcs=2, use_checkpoint=False, **kwargs):
        super(VisionTransformer, self).__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if output_cls_token:
            assert with_cls_token, 'with_cls_token must be True if set output_cls_token to True.'
        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.use_checkpoint = use_checkpoint
        # Image to Patch Embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=norm_cfg if patch_norm else None,
        )
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True
                )
            )
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = BuildNormalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))
    '''initialize backbone'''
    def initweights(self, vit_type='jx_vit_large_p16_384', pretrained_style='timm', pretrained_model_path=''):
        assert pretrained_style in ['timm', 'mmcls']
        if pretrained_model_path:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(model_urls[vit_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if pretrained_style == 'timm':
            state_dict = self.vitconvert(state_dict)
        if 'pos_embed' in state_dict.keys():
            if self.pos_embed.shape != state_dict['pos_embed'].shape:
                h, w = self.img_size
                pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                state_dict['pos_embed'] = self.resizeposembed(
                    state_dict['pos_embed'], 
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size), 
                    self.interpolate_mode
                )
        self.load_state_dict(state_dict, False)
    '''vit convert'''
    @staticmethod
    def vitconvert(ckpt):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('head'): 
                continue
            if k.startswith('norm'): 
                new_k = k.replace('norm.', 'ln1.')
            elif k.startswith('patch_embed'):
                if 'proj' in k: new_k = k.replace('proj', 'projection')
                else: new_k = k
            elif k.startswith('blocks'):
                if 'norm' in k: new_k = k.replace('norm', 'ln')
                elif 'mlp.fc1' in k: new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in k: new_k = k.replace('mlp.fc2', 'ffn.layers.1')
                elif 'attn.qkv' in k: new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
                elif 'attn.proj' in k: new_k = k.replace('attn.proj', 'attn.attn.out_proj')
                else: new_k = k
                new_k = new_k.replace('blocks.', 'layers.')
            else:
                new_k = k
            new_ckpt[new_k] = v
        return new_ckpt
    '''positiong embeding method'''
    def posembeding(self, patched_img, hw_shape, pos_embed):
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, 'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError('Unexpected shape of pos_embed, got {}.'.format(pos_embed.shape))
            pos_embed = self.resizeposembed(pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)
    '''resize pos_embed weights'''
    @staticmethod
    def resizeposembed(pos_embed, input_shpae, pos_shape, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    '''forward'''
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs), (self.patch_embed.DH, self.patch_embed.DW)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.posembeding(x, hw_shape, self.pos_embed)
        # Remove class token for transformer encoder input
        if not self.with_cls_token:
            x = x[:, 1:]
        outs = []
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.ln1(x)
            if i in self.out_indices:
                # Remove class token and reshape token for decoder head
                if self.with_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
                if self.output_cls_token: out = [out, x[:, 0]]
                outs.append(out)
        return tuple(outs)


'''build vision transformer'''
def BuildVisionTransformer(vit_type='jx_vit_large_p16_384', **kwargs):
    # assert whether support
    supported_vits = {
        'jx_vit_large_p16_384': {
            'img_size': 384,
            'patch_size': 16,
            'embed_dims': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.1,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.,
            'with_cls_token': True,
            'output_cls_token': False,
            'patch_norm': False,
            'final_norm': False,
            'num_fcs': 2,
        }
    }
    assert vit_type in supported_vits, 'unspport the vit_type %s...' % vit_type
    # parse args
    default_args = {
        'in_channels': 3,
        'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'layernorm', 'opts': {'eps': 1e-6}},
        'act_cfg': {'type': 'gelu', 'opts': {}},
        'interpolate_mode': 'bilinear',
        'pretrained': True,
        'pretrained_style': 'timm',
        'pretrained_model_path': '',
        'use_checkpoint': False,
    }
    default_args.update(supported_vits[vit_type])
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain the instanced vit
    vit_args = default_args.copy()
    model = VisionTransformer(**vit_args)
    # load weights of pretrained model
    if default_args['pretrained']:
        model.initweights(vit_type, default_args['pretrained_style'], default_args['pretrained_model_path'])
    # return the model
    return model