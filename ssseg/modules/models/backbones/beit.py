'''
Function:
    Implementation of BEiT
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from scipy import interpolate
from .vit import TransformerEncoderLayer as VisionTransformerEncoderLayer
from .bricks import BuildNormalization, PatchEmbed, BuildDropout, truncnormal


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'beit_base_patch16_224_pt22k_ft22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_beit/beit_base_patch16_224_pt22k_ft22k.pth',
    'beit_large_patch16_224_pt22k_ft22k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_beit/beit_large_patch16_224_pt22k_ft22k.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {
    'beit_base_patch16_224_pt22k_ft22k': {},
    'beit_large_patch16_224_pt22k_ft22k': {
        'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
        'qv_bias': True, 'init_values': 1e-6, 'drop_path_rate': 0.2, 'out_indices': [7, 11, 15, 23]
    },
}


'''BEiTAttention'''
class BEiTAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, window_size, bias='qv_bias', qk_scale=None, attn_drop_rate=0., proj_drop_rate=0.):
        super(BEiTAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.bias = bias
        self.scale = qk_scale or head_embed_dims**-0.5
        qkv_bias = bias
        if bias == 'qv_bias':
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))
            qkv_bias = False
        self.window_size = window_size
        self.initrelposembedding()
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        truncnormal(self.relative_position_bias_table, std=0.02)
    '''initrelposembedding'''
    def initrelposembedding(self):
        Wh, Ww = self.window_size
        # cls to token & token 2 cls & cls to cls
        self.num_relative_distance = (2 * Wh - 1) * (2 * Ww - 1) + 3
        # relative_position_bias_table shape is (2*Wh-1 * 2*Ww-1 + 3, nH)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, self.num_heads))
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = torch.zeros(size=(Wh * Ww + 1, ) * 2, dtype=relative_coords.dtype)
        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer('relative_position_index', relative_position_index)
    '''forward'''
    def forward(self, x):
        B, N, C = x.shape
        if self.bias == 'qv_bias':
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            Wh = self.window_size[0]
            Ww = self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(Wh * Ww + 1, Wh * Ww + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


'''BEiTTransformerEncoderLayer'''
class BEiTTransformerEncoderLayer(VisionTransformerEncoderLayer):
    def __init__(self, embed_dims, num_heads, feedforward_channels, attn_drop_rate=0., drop_path_rate=0., num_fcs=2, bias='qv_bias', 
                 act_cfg=None, norm_cfg=None, window_size=None, attn_cfg=dict(), ffn_cfg=dict(add_identity=False), init_values=None):
        super(BEiTTransformerEncoderLayer, self).__init__(
            embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels, attn_drop_rate=attn_drop_rate,
            drop_path_rate=0., drop_rate=0., num_fcs=num_fcs, qkv_bias=bias, act_cfg=act_cfg, norm_cfg=norm_cfg, attn_cfg=dict(), 
            ffn_cfg=ffn_cfg
        )
        dropout_cfg = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = BuildDropout(dropout_cfg) if dropout_cfg else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones((embed_dims)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((embed_dims)), requires_grad=True)
        attn_cfg.update(dict(
            window_size=window_size, qk_scale=None, embed_dims=embed_dims, num_heads=num_heads,
            attn_drop_rate=attn_drop_rate, proj_drop_rate=0., bias=bias,
        ))
        self.attn = BEiTAttention(**attn_cfg)
    '''forward'''
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.ln1(x)))
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.ln2(x)))
        return x


'''BEiT'''
class BEiT(nn.Module):
    def __init__(self, structure_type, img_size=(640, 640), patch_size=16, in_channels=3, embed_dims=768, num_layers=12, num_heads=12, mlp_ratio=4,
                 out_indices=(3, 5, 7, 11), qv_bias=True, attn_drop_rate=0., drop_path_rate=0.1, norm_cfg={'type': 'LayerNorm', 'eps': 1e-6}, 
                 act_cfg={'type': 'GELU'}, patch_norm=False, final_norm=False, num_fcs=2, init_values=0.1, pretrained=True, pretrained_model_path=''):
        super(BEiT, self).__init__()
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # set attributes
        self.structure_type = structure_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.qv_bias = qv_bias
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.patch_norm = patch_norm
        self.final_norm = final_norm
        self.num_fcs = num_fcs
        self.init_values = init_values
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        self.window_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.patch_shape = self.window_size
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        # set modules
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.buildpatchembedding()
        self.buildlayers()
        if final_norm:
            self.norm1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''buildpatchembedding'''
    def buildpatchembedding(self):
        self.patch_embed = PatchEmbed(
            in_channels=self.in_channels, embed_dims=self.embed_dims,
            kernel_size=self.patch_size, stride=self.patch_size, padding=0,
            norm_cfg=self.norm_cfg if self.patch_norm else None
        )
    '''buildlayers'''
    def buildlayers(self):
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(BEiTTransformerEncoderLayer(
                embed_dims=self.embed_dims, num_heads=self.num_heads, feedforward_channels=self.mlp_ratio * self.embed_dims,
                attn_drop_rate=self.attn_drop_rate, drop_path_rate=dpr[i], num_fcs=self.num_fcs, bias='qv_bias' if self.qv_bias else False,
                act_cfg=self.act_cfg, norm_cfg=self.norm_cfg, window_size=self.window_size, init_values=self.init_values
            ))
    '''geometricsequenceinterpolation'''
    def geometricsequenceinterpolation(self, src_size, dst_size, sequence, num):
        def geometricprogression(a, r, n):
            return a * (1.0 - r**n) / (1.0 - r)
        # here is a binary function.
        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometricprogression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        # the position of each interpolated point is determined by the ratio obtained by dichotomy.
        dis, cur = [], 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q**(i + 1)
        r_ids = [-d for d in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        # interpolation functions are being executed and called.
        new_sequence = []
        for i in range(num):
            z = sequence[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            new_sequence.append(torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(sequence))
        new_sequence = torch.cat(new_sequence, dim=-1)
        return new_sequence
    '''resizerelposembed'''
    def resizerelposembed(self, checkpoint):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        all_keys = list(state_dict.keys())
        for key in all_keys:
            if 'relative_position_index' in key:
                state_dict.pop(key)
            if 'relative_position_bias_table' in key:
                rel_pos_bias = state_dict[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.state_dict()[key].size()
                dst_patch_shape = self.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens)**0.5)
                dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                    new_rel_pos_bias = self.geometricsequenceinterpolation(src_size, dst_size, rel_pos_bias, num_attn_heads)
                    new_rel_pos_bias = torch.cat((new_rel_pos_bias, extra_tokens), dim=0)
                    state_dict[key] = new_rel_pos_bias
        return state_dict
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type='beit_base_patch16_224_pt22k_ft22k', pretrained_model_path=''):
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
        state_dict = self.beitconvert(state_dict)
        state_dict = self.resizerelposembed(state_dict)
        self.load_state_dict(state_dict, strict=False)
    '''beitconvert'''
    @staticmethod
    def beitconvert(ckpt):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('blocks'):
                new_key = k.replace('blocks', 'layers')
                if 'norm' in new_key:
                    new_key = new_key.replace('norm', 'ln')
                elif 'mlp.fc1' in new_key:
                    new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in new_key:
                    new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
                new_ckpt[new_key] = v
            elif k.startswith('patch_embed'):
                new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
                new_ckpt[new_key] = v
            else:
                new_key = k
                new_ckpt[new_key] = v
        return new_ckpt
    '''forward'''
    def forward(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)