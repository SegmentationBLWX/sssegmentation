'''
Function:
    Implementation of Predictor
Author:
    Zhenchao Jin
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer


'''PositionEmbeddingSine'''
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, apply_normalize=False, scale=None, eps=1e-6):
        super(PositionEmbeddingSine, self).__init__()
        assert not ((scale is not None) and (not apply_normalize))
        self.eps = eps
        self.temperature = temperature
        self.num_pos_feats = num_pos_feats
        self.apply_normalize = apply_normalize
        if scale is None: scale = 2 * math.pi
        self.scale = scale
    '''forward'''
    def forward(self, x, mask=None):
        if mask is None: mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.apply_normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


'''MLP'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    '''forward'''
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


'''Predictor'''
class Predictor(nn.Module):
    def __init__(self, in_channels, mask_classification, num_classes, hidden_dim, num_queries, nheads, dropout, dim_feedforward, enc_layers, dec_layers, pre_norm, deep_supervision, mask_dim, enforce_input_project, norm_cfg=None, act_cfg=None):
        super(Predictor, self).__init__()
        self.num_queries = num_queries
        self.in_channels = in_channels
        self.aux_loss = deep_supervision
        self.mask_classification = mask_classification
        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, apply_normalize=True)
        # transformer
        self.transformer = Transformer(
            d_model=hidden_dim, 
            nhead=nheads, 
            num_encoder_layers=enc_layers, 
            num_decoder_layers=dec_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            norm_before=pre_norm, 
            return_intermediate_dec=deep_supervision,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )
        hidden_dim = self.transformer.d_model
        # query embed
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # input project
        if in_channels != hidden_dim or enforce_input_project:
            import fvcore.nn.weight_init as weight_init
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        # output FFNs
        if self.mask_classification: self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # mask embed
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    '''forward'''
    def forward(self, x, mask_features):
        # feed to transformer
        hs, memory = self.transformer(self.input_proj(x), None, self.query_embed.weight, self.pe_layer(x))
        # mask classification
        outputs = {}
        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            outputs.update({'pred_logits': self.class_embed(hs)[-1]})
        # auxiliary layer
        if self.aux_loss:
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum('lbqc,bchw->lbqhw', mask_embed, mask_features)
            outputs['pred_masks'] = outputs_seg_masks[-1]
            outputs['aux_outputs'] = self.setauxloss(outputs_class if self.mask_classification else None, outputs_seg_masks)
        else:
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
            out['pred_masks'] = outputs_seg_masks
        return outputs
    '''setauxloss'''
    @torch.jit.unused
    def setauxloss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]