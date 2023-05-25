'''
Function:
    Implementation of nn.Transformer with modifications
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....backbones import BuildActivation, BuildNormalization


'''TransformerEncoderLayer'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_cfg=None, act_cfg=None, norm_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm_before = norm_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # linear
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # norm
        self.norm1 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.norm2 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        # act
        self.activation = BuildActivation(act_cfg)
    '''with pos embed'''
    def withposembed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    '''norm_before=False forward'''
    def normafterforward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.withposembed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    '''norm_before=True forward'''
    def normbeforeforward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.withposembed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    '''forward'''
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.norm_before: return self.normbeforeforward(src, src_mask, src_key_padding_mask, pos)
        return self.normafterforward(src, src_mask, src_key_padding_mask, pos)


'''TransformerEncoder'''
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.norm = norm
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
    '''forward'''
    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None: output = self.norm(output)
        return output


'''TransformerDecoderLayer'''
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_cfg=None, act_cfg=None, norm_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.norm_before = norm_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # linear
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # norm
        self.norm1 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.norm2 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.norm3 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # act
        self.activation = BuildActivation(act_cfg)
    '''with pos embed'''
    def withposembed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    '''norm_before=False forward'''
    def normafterforward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        q = k = self.withposembed(tgt, query_pos)
        tgt = tgt + self.dropout1(self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        tgt = self.norm1(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(self.withposembed(tgt, query_pos), self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0])
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))
        tgt = self.norm3(tgt)
        return tgt
    '''norm_before=True forward'''
    def normbeforeforward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt_norm = self.norm1(tgt)
        q = k = self.withposembed(tgt_norm, query_pos)
        tgt = tgt + self.dropout1(self.self_attn(q, k, value=tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        tgt_norm = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(self.withposembed(tgt_norm, query_pos), self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0])
        tgt_norm = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt_norm)))))
        return tgt
    '''forward'''
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.norm_before:
            return self.normbeforeforward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.normafterforward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


'''TransformerDecoder'''
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.norm = norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    '''forward'''
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        output, intermediate = tgt, []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate: intermediate.append(self.norm(output))
        if self.norm is not None: output = self.norm(output)
        if self.return_intermediate: return torch.stack(intermediate)
        return output.unsqueeze(0)


'''Transformer'''
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, norm_cfg=None, act_cfg=None, norm_before=False, return_intermediate_dec=False):
        super(Transformer, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_cfg, act_cfg, norm_before)
        encoder_norm = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg) if norm_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_cfg, act_cfg, norm_before)
        decoder_norm = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        # reset parameters
        self.resetparameters()
    '''reset parameters'''
    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    '''forward'''
    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None: mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)