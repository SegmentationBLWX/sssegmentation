'''
Function:
    define the semantic-level context module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''semantic-level context module'''
class SemanticLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, **kwargs):
        super(SemanticLevelContext, self).__init__()
        norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(norm_cfg['type'], (feats_channels, norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            )
    '''forward'''
    def forward(self, x, preds, feats_il):
        inputs = x
        batch_size, num_channels, h, w = x.size()
        num_classes = preds.size(1)
        feats_sl = torch.zeros(batch_size, h*w, num_channels).type_as(x)
        for batch_idx in range(batch_size):
            # (C, H, W), (num_classes, H, W) --> (H*W, C), (H*W, num_classes)
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
            # (H*W, )
            argmax = preds_iter.argmax(1)
            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0: continue
                feats_iter_cls = feats_iter[mask]
                preds_iter_cls = preds_iter[:, clsid][mask]
                weight = F.softmax(preds_iter_cls, dim=0)
                feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                feats_iter_cls = feats_iter_cls.sum(0)
                feats_sl[batch_idx][mask] = feats_iter_cls
        feats_sl = feats_sl.reshape(batch_size, h, w, num_channels)
        feats_sl = feats_sl.permute(0, 3, 1, 2).contiguous()
        feats_sl = self.correlate_net(inputs, feats_sl)
        if hasattr(self, 'bottleneck'):
            feats_sl = self.bottleneck(torch.cat([feats_il, feats_sl], dim=1))
        return feats_sl