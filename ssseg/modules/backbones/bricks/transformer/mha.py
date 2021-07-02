'''
Function:
    Multi-head Attention Module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..dropout import BuildDropout


'''Multi-head Attention'''
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_cfg=None, batch_first=False, **kwargs):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        if self.batch_first:
            def bnctonbc(forward):
                def forwardwrapper(**kwargs):
                    convert_keys = ('key', 'query', 'value')
                    for key in kwargs.keys():
                        if key in convert_keys:
                            kwargs[key] = kwargs[key].transpose(0, 1)
                    attn_output, attn_output_weights = forward(**kwargs)
                    return attn_output.transpose(0, 1), attn_output_weights
                return forwardwrapper
            self.attn.forward = bnctonbc(self.attn.forward)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = BuildDropout(dropout_cfg['type'], **dropout_cfg['opts']) if dropout_cfg else nn.Identity()
    '''forward'''
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, **kwargs):
        if key is None: key = query
        if value is None: value = key
        if identity is None: identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return identity + self.dropout_layer(self.proj_drop(out))