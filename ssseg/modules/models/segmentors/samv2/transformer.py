'''
Function:
    Implementation of transformer-related modules
Author:
    Zhenchao Jin
'''
import math
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ...backbones.hiera import MLP
from ....utils import BaseModuleBuilder
from .misc import getsdpasettings, applyrotaryenc, computeaxialcis
warnings.simplefilter(action="ignore", category=FutureWarning)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = getsdpasettings()


'''TwoWayTransformer'''
class TwoWayTransformer(nn.Module):
    def __init__(self, depth, embedding_dim, num_heads, mlp_dim, act_cfg={'type': 'ReLU'}, attention_downsample_rate=2):
        super(TwoWayTransformer, self).__init__()
        # set attributes
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        # build layers
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(
                embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, act_cfg=act_cfg, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=(i == 0),
            ))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    '''forward'''
    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # prepare queries
        queries = point_embedding
        keys = image_embedding
        # apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        # apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        # return
        return queries, keys


'''TwoWayAttentionBlock'''
class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, act_cfg={'type': 'ReLU'}, attention_downsample_rate=2, skip_first_layer_pe=False):
        super(TwoWayAttentionBlock, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act_cfg=act_cfg)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe
    '''forward'''
    def forward(self, queries, keys, query_pe, key_pe):
        # self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        # cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        # mlp block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        # cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        # return
        return queries, keys


'''Attention'''
class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1, dropout=0.0, kv_in_dim=None):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout
    '''separateheads'''
    def separateheads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)
    '''recombineheads'''
    def recombineheads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)
    '''forward'''
    def forward(self, q, k, v):
        # input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # separate into heads
        q = self.separateheads(q, self.num_heads)
        k = self.separateheads(k, self.num_heads)
        v = self.separateheads(v, self.num_heads)
        dropout_p = self.dropout_p if self.training else 0.0
        # attention
        with torch.backends.cuda.sdp_kernel(enable_flash=USE_FLASH_ATTN, enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON, enable_mem_efficient=OLD_GPU):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self.recombineheads(out)
        out = self.out_proj(out)
        # return
        return out


'''RoPEAttention'''
class RoPEAttention(Attention):
    def __init__(self, *args, rope_theta=10000.0, rope_k_repeat=False, feat_sizes=(32, 32), **kwargs):
        super(RoPEAttention, self).__init__(*args, **kwargs)
        self.compute_cis = partial(computeaxialcis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat
    '''forward'''
    def forward(self, q, k, v, num_k_exclude_rope=0):
        # input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # separate into heads
        q = self.separateheads(q, self.num_heads)
        k = self.separateheads(k, self.num_heads)
        v = self.separateheads(v, self.num_heads)
        # apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = applyrotaryenc(q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)
        dropout_p = self.dropout_p if self.training else 0.0
        # attention
        with torch.backends.cuda.sdp_kernel(enable_flash=USE_FLASH_ATTN, enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON, enable_mem_efficient=OLD_GPU):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self.recombineheads(out)
        out = self.out_proj(out)
        # return
        return out


'''AttentionBuilder'''
class AttentionBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'Attention': Attention, 'RoPEAttention': RoPEAttention
    }
    '''build'''
    def build(self, pe_cfg):
        return super().build(pe_cfg)


'''BuildAttention'''
BuildAttention = AttentionBuilder().build