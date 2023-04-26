'''
Function:
    Implementation of MultiheadAttention
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..dropout import BuildDropout


'''MultiheadAttention'''
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_cfg=None, batch_first=False, **kwargs):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = BuildDropout(dropout_cfg) if dropout_cfg else nn.Identity()
    '''forward'''
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None):
        if key is None: key = query
        if value is None: value = key
        if identity is None: identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape: key_pos = query_pos
        if query_pos is not None: query = query + query_pos
        if key_pos is not None: key = key + key_pos
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        if self.batch_first: out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))