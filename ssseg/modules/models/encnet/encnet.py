'''
Function:
    Implementation of ENCNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .contextencoding import ContextEncoding


'''ENCNet'''
class ENCNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(ENCNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build encoding
        # --base structurs
        encoding_cfg = cfg['encoding']
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoding_cfg['in_channels_list'][-1], encoding_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (encoding_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.enc_module = ContextEncoding(
            in_channels=encoding_cfg['out_channels'],
            num_codes=encoding_cfg['num_codes'],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # --extra structures
        extra_cfg = encoding_cfg['extra']
        if extra_cfg['add_lateral']:
            self.lateral_convs = nn.ModuleList()
            for in_channels in encoding_cfg['in_channels_list'][:-1]:
                self.lateral_convs.append(
                    nn.Conv2d(in_channels, encoding_cfg['out_channels'], kernel_size=1, stride=1, padding=0),
                    BuildNormalization(norm_cfg['type'], (encoding_cfg['out_channels'], norm_cfg['opts'])),
                    BuildActivation(act_cfg['type'], **act_cfg['opts']),
                )
            self.fusion = nn.Sequential(
                nn.Conv2d(len(encoding_cfg['in_channels_list']) * encoding_cfg['out_channels'], encoding_cfg['out_channels'], kernel_size=3, stride=1, padding=1),
                BuildNormalization(norm_cfg['type'], (encoding_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            )
        if extra_cfg['use_se_loss']:
            self.se_layer = nn.Linear(encoding_cfg['out_channels'], cfg['num_classes'])
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(auxiliary_cfg['dropout']),
            nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to context encoding
        feats = self.bottleneck(outputs[-1])
        if hasattr(self, 'lateral_convs'):
            lateral_outs = [
                F.interpolate(lateral_conv(outputs[idx]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners) for idx, lateral_conv in enumerate(self.lateral_convs)
            ]
            feats = self.fusion(torch.cat([feats, *lateral_outs], dim=1))
        encode_feats, feats = self.enc_module(feats)
        if hasattr(self, 'se_layer'):
            preds_se = self.se_layer(encode_feats)
        # feed to decoder
        preds = self.decoder(feats)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(outputs[-2])
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            if hasattr(self, 'se_layer'):
                predictions = {'loss_cls': preds, 'loss_se': preds_se, 'loss_aux': preds_aux}
            else:
                predictions = {'loss_cls': preds, 'loss_aux': preds_aux}
            return self.calculatelosses(
                predictions=predictions, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        layers = {
            'backbone_net': self.backbone_net,
            'bottleneck': self.bottleneck,
            'enc_module': self.enc_module,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }
        if hasattr(self, 'lateral_convs'): layers['lateral_convs'] = self.lateral_convs
        if hasattr(self, 'fusion'): layers['fusion'] = self.fusion
        if hasattr(self, 'se_layer'): layers['se_layer'] = self.se_layer
        return layers
    '''convert to onehot labels'''
    def onehot(self, labels, num_classes):
        batch_size = labels.size(0)
        labels_onehot = labels.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = labels[i].float().histc(bins=num_classes, min=0, max=num_classes-1)
            labels_onehot[i] = hist > 0
        return labels_onehot