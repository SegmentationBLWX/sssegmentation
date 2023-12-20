'''
Function:
    Implementation of TwoWayTransformer
Author:
    Zhenchao Jin
'''
import math
import torch
import torch.nn as nn
from ...backbones.samvit import MLPBlock


'''Attention'''
class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
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
        # attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # get output
        out = attn @ v
        out = self.recombineheads(out)
        out = self.out_proj(out)
        # return out
        return out


'''TwoWayAttentionBlock'''
class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, act_cfg={'type': 'ReLU'}, attention_downsample_rate=2, skip_first_layer_pe=False):
        super(TwoWayAttentionBlock, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, act_cfg)
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
        # MLP block
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


'''TwoWayTransformer'''
class TwoWayTransformer(nn.Module):
    def __init__(self, depth, embedding_dim, num_heads, mlp_dim, act_cfg={'type': 'ReLU'}, attention_downsample_rate=2):
        super(TwoWayTransformer, self).__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(
                embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, act_cfg=act_cfg, 
                attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=(i == 0),
            ))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    '''forward'''
    def forward(self, image_embedding, image_pe, point_embedding):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
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