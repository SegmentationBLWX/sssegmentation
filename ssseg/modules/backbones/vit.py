'''
Function:
    Implementation of VIT
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bricks import BuildNormalization, BuildActivation, DropPath, truncnormal


'''MLP layer for Encoder block'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = BuildActivation(act_cfg['type'], **act_cfg['opts'])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    '''forward'''
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''Attention layer for Encoder block'''
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    '''forward'''
    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


'''Implements encoder block with residual connection.'''
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., proj_drop=0., 
                 drop_path=0., act_cfg=None, norm_cfg=None):
        super(Block, self).__init__()
        self.norm1 = BuildNormalization(norm_cfg['type'], (dim, norm_cfg['opts']))
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = BuildNormalization(norm_cfg['type'], (dim, norm_cfg['opts']))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop)
    '''forward'''
    def forward(self, x):
        out = x + self.drop_path(self.attn(self.norm1(x)))
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out


'''Image to Patch Embedding.'''
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        if isinstance(img_size, int): self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple): self.img_size = img_size
        else: raise TypeError('img_size must be type of int or tuple')
        h, w = self.img_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    '''forward'''
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


'''Vision transformer backbone.'''
class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
                 out_indices=11, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_cfg=None, act_cfg=None, norm_eval=False, final_norm=False, with_cls_token=True, interpolate_mode='bicubic', **kwargs):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.with_cls_token = with_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if isinstance(out_indices, int):
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dpr[i],
                attn_drop=attn_drop_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg) for i in range(depth)
        ])
        self.interpolate_mode = interpolate_mode
        self.final_norm = final_norm
        if final_norm:
            self.norm = BuildNormalization(norm_cfg['type'], (embed_dim, norm_cfg['opts']))
        self.norm_eval = norm_eval
    '''init weights'''
    def initweights(self, pretrained_model_path=None):
        if (pretrained_model_path is not None) and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path)
            if 'state_dict' in checkpoint: 
                state_dict = checkpoint['state_dict']
            else: 
                state_dict = checkpoint
            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    h, w = self.img_size
                    pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resizeposembed(state_dict['pos_embed'], (h, w), (pos_size, pos_size), self.patch_size, self.interpolate_mode)
            self.load_state_dict(state_dict, False)
        else:
            from mmcv.utils.parrots_wrapper import _BatchNorm
            from mmcv.cnn import constant_init, kaiming_init, normal_init
            truncnormal(self.pos_embed, std=.02)
            truncnormal(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    truncnormal(m.weight, std=.02)
                    if m.bias is not None:
                        if 'mlp' in n:
                            normal_init(m.bias, std=1e-6)
                        else:
                            constant_init(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m.weight, mode='fan_in')
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1)
    '''Positiong embeding method.'''
    def posembeding(self, img, patched_img, pos_embed):
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, 'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError('Unexpected shape of pos_embed, got {}.'.format(pos_embed.shape))
            pos_embed = self.resizeposembed(pos_embed, img.shape[2:], (pos_h, pos_w), self.patch_size, self.interpolate_mode)
        return self.pos_drop(patched_img + pos_embed)
    '''Resize pos_embed weights'''
    @staticmethod
    def resizeposembed(pos_embed, input_shpae, pos_shape, patch_size, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        input_h, input_w = input_shpae
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=[input_h // patch_size, input_w // patch_size], align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    '''forward'''
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        x = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.posembeding(inputs, x, self.pos_embed)
        if not self.with_cls_token:
            x = x[:, 1:]
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                if self.final_norm: x = self.norm(x)
            if i in self.out_indices:
                if self.with_cls_token: out = x[:, 1:]
                else: out = x
                batch_size, _, C = out.shape
                out = out.reshape(batch_size, inputs.shape[2] // self.patch_size, inputs.shape[3] // self.patch_size, C).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)
    '''set train'''
    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


'''build vision transformer'''
def BuildVisionTransformer(vit_type=None, **kwargs):
    # assert whether support
    assert vit_type is None
    # parse args
    default_args = {
        'img_size': (224, 224),
        'patch_size': 16,
        'in_channels': 3,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'out_indices': 11,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.,
        'norm_cfg': {'type': 'layernorm', 'opts': {'eps': 1e-6}},
        'act_cfg': {'type': 'gelu', 'opts': {}},
        'norm_eval': False,
        'final_norm': False,
        'with_cls_token': True,
        'interpolate_mode': 'bicubic',
        'pretrained': True,
        'pretrained_model_path': '',
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain the instanced vit
    vit_args = default_args.copy()
    model = VisionTransformer(**vit_args)
    # load weights of pretrained model
    if default_args['pretrained']:
        model.initweights(default_args['pretrained_model_path'])
    else:
        model.initweights('')
    # return the model
    return model