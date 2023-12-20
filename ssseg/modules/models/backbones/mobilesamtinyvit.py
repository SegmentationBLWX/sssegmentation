'''
Function:
    Implementation of MobileSAMTinyViT
Author:
    Zhenchao Jin
'''
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .samvit import LayerNorm2d
from .bricks import BuildActivation
from ...utils import loadpretrainedweights
from .bricks.dropout.droppath import DropPath


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'tiny_vit_5m_1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_5m_1k.pth',
    'tiny_vit_11m_1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_11m_1k.pth',
    'tiny_vit_21m_1k': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_21m_1k.pth',
    'tiny_vit_5m_22k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_5m_22k_distill.pth',
    'tiny_vit_11m_22k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_11m_22k_distill.pth',
    'tiny_vit_21m_22k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_21m_22k_distill.pth',
    'tiny_vit_5m_22kto1k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_5m_22kto1k_distill.pth',
    'tiny_vit_11m_22kto1k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_11m_22kto1k_distill.pth',
    'tiny_vit_21m_22kto1k_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_21m_22kto1k_distill.pth',
    'tiny_vit_21m_22kto1k_384_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_21m_22kto1k_384_distill.pth',
    'tiny_vit_21m_22kto1k_512_distill': 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/tiny_vit_21m_22kto1k_512_distill.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {
    'tiny_vit_5m_22kto1k_distill': {
        'embed_dims': [64, 128, 160, 320], 'depths': [2, 2, 6, 2], 'num_heads': [2, 4, 5, 10], 'window_sizes': [7, 7, 14, 7], 
    },
    'tiny_vit_11m_22kto1k_distill': {
        'embed_dims': [64, 128, 256, 448], 'depths': [2, 2, 6, 2], 'num_heads': [2, 4, 8, 14], 'window_sizes': [7, 7, 14, 7], 
    },
    'tiny_vit_21m_22kto1k_distill': {
        'embed_dims': [96, 192, 384, 576], 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 18], 'window_sizes': [7, 7, 14, 7], 
    },
    'tiny_vit_21m_22kto1k_384_distill': {
        'embed_dims': [96, 192, 384, 576], 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 18], 'window_sizes': [12, 12, 24, 12], 
    },
    'tiny_vit_21m_22kto1k_512_distill': {
        'embed_dims': [96, 192, 384, 576], 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 18], 'window_sizes': [16, 16, 32, 16], 
    },
}


'''Conv2dBN'''
class Conv2dBN(nn.Sequential):
    def __init__(self, in_chans, out_chans, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dBN, self).__init__()
        self.add_module('c', nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chans))


'''MBConv'''
class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, act_cfg={'type': 'GELU'}, drop_path=0.0):
        super(MBConv, self).__init__()
        # set attributes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        # build layers
        self.conv1 = Conv2dBN(in_chans, self.hidden_chans, kernel_size=1, stride=1, padding=0)
        self.act1 = BuildActivation(act_cfg=act_cfg)
        self.conv2 = Conv2dBN(self.hidden_chans, self.hidden_chans, kernel_size=3, stride=1, padding=1, groups=self.hidden_chans)
        self.act2 = BuildActivation(act_cfg=act_cfg)
        self.conv3 = Conv2dBN(self.hidden_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.act3 = BuildActivation(act_cfg=act_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    '''forward'''
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


'''ConvLayer'''
class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, act_cfg={'type': 'GELU'}, drop_path=0., downsample=None, use_checkpoint=False, out_dim=None, conv_expand_ratio=4.):
        super(ConvLayer, self).__init__()
        # set attributes
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, act_cfg, drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, act_cfg=act_cfg)
        else:
            self.downsample = None
    '''forward'''
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


'''MLP'''
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg={'type': 'GELU'}, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = BuildActivation(act_cfg=act_cfg)
        self.drop = nn.Dropout(drop)
    '''forward'''
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''PatchEmbed'''
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, act_cfg={'type': 'GELU'}):
        super(PatchEmbed, self).__init__()
        img_size = self.totuple(resolution)
        # set attributes
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        # build seq
        self.seq = nn.Sequential(
            Conv2dBN(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            BuildActivation(act_cfg=act_cfg),
            Conv2dBN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        )
    '''forward'''
    def forward(self, x):
        return self.seq(x)
    '''totuple'''
    @staticmethod
    def totuple(x):
        if isinstance(x, int): return (x, x)
        assert isinstance(x, tuple) and (len(x) == 2)
        return x


'''PatchMerging'''
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, act_cfg={'type': 'GELU'}):
        super(PatchMerging, self).__init__()
        # set attributes
        self.dim = dim
        self.act = BuildActivation(act_cfg=act_cfg)
        self.out_dim = out_dim
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        # build layers
        self.conv1 = Conv2dBN(dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2dBN(out_dim, out_dim, kernel_size=3, stride=1 if (out_dim == 320 or out_dim == 448 or out_dim == 576) else 2, padding=1, groups=out_dim)
        self.conv3 = Conv2dBN(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
    '''forward'''
    def forward(self, x):
        if x.ndim == 3: x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


'''Attention'''
class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=(14, 14)):
        super(Attention, self).__init__()
        # assert
        assert isinstance(resolution, tuple) and len(resolution) == 2
        # set attributes
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        # build layers
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(self.dh, dim)
        self.qkv = nn.Linear(dim, self.dh + (key_dim * num_heads) * 2)
        idxs, attention_offsets = [], {}
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(len(points), len(points)), persistent=False)
    '''train'''
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.register_buffer('ab', self.attention_biases[:, self.attention_bias_idxs], persistent=False)
    '''forward'''
    def forward(self, x):
        batch_size, num_pixels = x.shape[:2]
        # obtain qkv
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.view(batch_size, num_pixels, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # feed to attn and proj
        attn = (
            (q @ k.transpose(-2, -1)) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_pixels, self.dh)
        x = self.proj(x)
        # return
        return x


'''TinyViTBlock'''
class TinyViTBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., drop=0., drop_path=0., local_conv_size=3, act_cfg={'type': 'GELU'}):
        super(TinyViTBlock, self).__init__()
        # assert
        assert window_size > 0, 'window_size must be greater than 0'
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        # set attributes
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        # build layers
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(dim, dim // num_heads, num_heads, attn_ratio=1, resolution=PatchEmbed.totuple(window_size))
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_cfg=act_cfg, drop=drop)
        self.local_conv = Conv2dBN(dim, dim, kernel_size=local_conv_size, stride=1, padding=local_conv_size // 2, groups=dim)
    '''forward'''
    def forward(self, x):
        x_identity = x
        batch_size, num_pixels, num_channels = x.shape
        # assert
        assert x.shape[1] == self.input_resolution[0] * self.input_resolution[1], 'input feature has wrong size'
        # feed to attn
        if self.input_resolution[0] == self.window_size and self.input_resolution[1] == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(batch_size, self.input_resolution[0], self.input_resolution[1], num_channels)
            pad_b = (self.window_size - self.input_resolution[0] % self.window_size) % self.window_size
            pad_r = (self.window_size - self.input_resolution[1] % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding: x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = self.input_resolution[0] + pad_b, self.input_resolution[1] + pad_r
            nH, nW = pH // self.window_size, pW // self.window_size
            # window partition
            x = x.view(batch_size, nH, self.window_size, nW, self.window_size, num_channels).transpose(2, 3).reshape(batch_size * nH * nW, self.window_size * self.window_size, num_channels)
            x = self.attn(x)
            # window reverse
            x = x.view(batch_size, nH, nW, self.window_size, self.window_size, num_channels).transpose(2, 3).reshape(batch_size, pH, pW, num_channels)
            if padding: x = x[:, :self.input_resolution[0], :self.input_resolution[1]].contiguous()
            x = x.view(batch_size, num_pixels, num_channels)
        # feed to other layers
        x = x_identity + self.drop_path(x)
        x = x.transpose(1, 2).reshape(batch_size, num_channels, self.input_resolution[0], self.input_resolution[1])
        x = self.local_conv(x)
        x = x.view(batch_size, num_channels, num_pixels).transpose(1, 2)
        x = x + self.drop_path(self.mlp(x))
        # return
        return x


'''BasicLayer'''
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., drop=0., drop_path=0., downsample=None, use_checkpoint=False, local_conv_size=3, act_cfg={'type': 'GELU'}, out_dim=None):
        super(BasicLayer, self).__init__()
        # set attributes
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        # build blocks
        self.blocks = nn.ModuleList([TinyViTBlock(
            dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, drop=drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, local_conv_size=local_conv_size, act_cfg=act_cfg,
        ) for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, act_cfg=act_cfg)
        else:
            self.downsample = None
    '''forward'''
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


'''MobileSAMTinyViT'''
class MobileSAMTinyViT(nn.Module):
    def __init__(self, structure_type, img_size=224, in_chans=3, embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_sizes=[7, 7, 14, 7], 
                 mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3, act_cfg={'type': 'GELU'}, pretrained=False, 
                 pretrained_model_path=''):
        super(MobileSAMTinyViT, self).__init__()
        # build patch_embed
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, act_cfg=act_cfg)
        # set attributes
        self.depths = depths
        self.img_size=img_size
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.embed_dims = embed_dims
        self.num_layers = len(depths)
        self.window_sizes = window_sizes
        self.use_checkpoint = use_checkpoint
        self.drop_path_rate = drop_path_rate
        self.local_conv_size = local_conv_size
        self.mbconv_expand_ratio = mbconv_expand_ratio
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # build layers
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[layer_idx], depth=depths[layer_idx], drop_path=dpr[sum(depths[:layer_idx]): sum(depths[:layer_idx+1])], downsample=PatchMerging if (layer_idx < self.num_layers - 1) else None,
                input_resolution=(patches_resolution[0] // (2 ** (layer_idx-1 if layer_idx == 3 else layer_idx)), patches_resolution[1] // (2 ** (layer_idx-1 if layer_idx == 3 else layer_idx))),
                use_checkpoint=use_checkpoint, out_dim=embed_dims[min(layer_idx + 1, len(embed_dims) - 1)], act_cfg=act_cfg,
            )
            if layer_idx == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(num_heads=num_heads[layer_idx], window_size=window_sizes[layer_idx], mlp_ratio=self.mlp_ratio, drop=drop_rate, local_conv_size=local_conv_size, **kwargs)
            self.layers.append(layer)
        # build neck
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 256, kernel_size=1, stride=1, padding=0, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(256),
        )
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x, return_interm_embeddings=False):
        x = self.patch_embed(x)
        interm_embeddings = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 1:
                interm_embeddings.append(x.view(x.shape[0], 64, 64, -1))
        x = x.view(x.shape[0], 64, 64, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        if return_interm_embeddings:
            return x, interm_embeddings
        return x