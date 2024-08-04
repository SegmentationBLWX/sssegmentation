'''
Function:
    Implementation of MemoryAttention
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from ...backbones import BuildActivation
from ..mask2former.transformers.misc import getclones
from .transformer import RoPEAttention, BuildAttention


'''MemoryAttentionLayer'''
class MemoryAttentionLayer(nn.Module):
    def __init__(self, act_cfg, cross_attention_cfg, d_model, dim_feedforward, dropout, pos_enc_at_attn, pos_enc_at_cross_attn_keys, pos_enc_at_cross_attn_queries, self_attention_cfg):
        super(MemoryAttentionLayer, self).__init__()
        # set attributes
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = BuildAttention(self_attention_cfg)
        self.cross_attn_image = BuildAttention(cross_attention_cfg)
        # build layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg=act_cfg)
        # pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
    '''forwardsa'''
    def forwardsa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt
    '''forwardca'''
    def forwardca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2, k=memory + pos if self.pos_enc_at_cross_attn_keys else memory, v=memory, **kwds)
        tgt = tgt + self.dropout2(tgt2)
        return tgt
    '''forward'''
    def forward(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
        # self attention, cross attention
        tgt = self.forwardsa(tgt, query_pos)
        tgt = self.forwardca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # mlp
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        # return
        return tgt


'''MemoryAttention'''
class MemoryAttention(nn.Module):
    def __init__(self, d_model, pos_enc_at_input, layer_cfg, num_layers, batch_first=True):
        super(MemoryAttention, self).__init__()
        self.d_model = d_model
        layer = MemoryAttentionLayer(**layer_cfg)
        self.layers = getclones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
    '''forward'''
    def forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]
        assert curr.shape[1] == memory.shape[1], "Batch size must be the same for curr and memory"
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            output = layer(tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos, **kwds)
        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
        return normed_output