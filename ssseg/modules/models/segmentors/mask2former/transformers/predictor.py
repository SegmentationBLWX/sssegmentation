'''
Function:
    Implementation of MultiScaleMaskedTransformerDecoder
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...maskformer.transformers.predictor import MLP
from ....backbones import PositionEmbeddingSine, BuildActivation, BuildNormalization


'''SelfAttentionLayer'''
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, act_cfg={'type': 'ReLU', 'inplace': True}, normalize_before=False):
        super(SelfAttentionLayer, self).__init__()
        # set attributes
        self.normalize_before = normalize_before
        # define layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg=act_cfg)
        # reset parameters
        self.resetparameters()
    '''resetparameters'''
    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    '''withposembed'''
    def withposembed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    '''forwardpost'''
    def forwardpost(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.withposembed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
    '''forwardpre'''
    def forwardpre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.withposembed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt
    '''forward'''
    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forwardpre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forwardpost(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


'''CrossAttentionLayer'''
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, act_cfg={'type': 'ReLU', 'inplace': True}, normalize_before=False):
        super(CrossAttentionLayer, self).__init__()
        # set attributes
        self.normalize_before = normalize_before
        # define layers
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg)
        # reset parameters
        self.resetparameters()
    '''resetparameters'''
    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    '''withposembed'''
    def withposembed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    '''forwardpost'''
    def forwardpost(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.withposembed(tgt, query_pos), key=self.withposembed(memory, pos), value=memory, 
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
    '''forwardpre'''
    def forwardpre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.withposembed(tgt2, query_pos), key=self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt
    '''forward'''
    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forwardpre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forwardpost(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


'''FFNLayer'''
class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, act_cfg={'type': 'ReLU', 'inplace': True}, normalize_before=False):
        super(FFNLayer, self).__init__()
        # set attributes
        self.normalize_before = normalize_before
        # implementation of feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = BuildActivation(act_cfg)
        # reset parameters
        self.resetparameters()
    '''resetparameters'''
    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    '''withposembed'''
    def withposembed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    '''forwardpost'''
    def forwardpost(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
    '''forwardpre'''
    def forwardpre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt
    '''forward'''
    def forward(self, tgt):
        if self.normalize_before:
            return self.forwardpre(tgt)
        return self.forwardpost(tgt)


'''MultiScaleMaskedTransformerDecoder'''
class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, pre_norm, mask_dim, enforce_input_project, mask_classification=True):
        super(MultiScaleMaskedTransformerDecoder, self).__init__()
        # assert
        assert mask_classification, 'only support mask classification model'
        import fvcore.nn.weight_init as weight_init
        self.mask_classification = mask_classification
        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, apply_normalize=True)
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    '''forward'''
    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []
        # disable mask, it does not affect performance
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        _, bs, _ = src[0].shape
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        predictions_class, predictions_mask = [], []
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forwardpredictionheads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](output, src[level_index], memory_mask=attn_mask, memory_key_padding_mask=None, pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
            # FFN
            output = self.transformer_ffn_layers[i](output)
            outputs_class, outputs_mask, attn_mask = self.forwardpredictionheads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        # assert and return outputs
        assert len(predictions_class) == self.num_layers + 1
        out = {
            'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1], 
            'aux_outputs': self.setauxloss(predictions_class if self.mask_classification else None, predictions_mask)
        }
        return out
    '''forwardpredictionheads'''
    def forwardpredictionheads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        # NOTE: prediction is of higher-resolution [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode='bilinear', align_corners=False)
        # must use bool type, if a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        # return
        return outputs_class, outputs_mask, attn_mask
    '''setauxloss'''
    @torch.jit.unused
    def setauxloss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]