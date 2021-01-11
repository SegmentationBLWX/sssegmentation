'''
Function:
    Implementation of DNLNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .dnlblock import DisentangledNonLocal2d


'''DNLNet'''
class DNLNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(DNLNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build disentangled non-local block
        dnl_cfg = cfg['dnl']
        self.conv_before_dnl = nn.Sequential(
            nn.Conv2d(dnl_cfg['in_channels'], dnl_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (dnl_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.dnl_block = DisentangledNonLocal2d(
            in_channels=dnl_cfg['out_channels'],
            reduction=dnl_cfg['reduction'],
            use_scale=dnl_cfg['use_scale'],
            mode=dnl_cfg['mode'],
            temperature=dnl_cfg['temperature'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        self.conv_after_dnl = nn.Sequential(
            nn.Conv2d(dnl_cfg['out_channels'], dnl_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (dnl_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
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
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to disentangled non-local block
        feats = self.conv_before_dnl(x4)
        feats = self.dnl_block(feats)
        feats = self.conv_after_dnl(feats)
        # feed to decoder
        feats = torch.cat([x4, feats], dim=1)
        preds = self.decoder(feats)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(x3)
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_aux': preds_aux}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'conv_before_dnl': self.conv_before_dnl,
            'dnl_block': self.dnl_block,
            'conv_after_dnl': self.conv_after_dnl,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }