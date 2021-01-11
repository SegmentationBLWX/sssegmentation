'''
Function:
    Implementation of Deeplabv3
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import ASPP
from ...backbones import *
from ..base import BaseModel


'''Deeplabv3'''
class Deeplabv3(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(Deeplabv3, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build aspp net
        aspp_cfg = {
            'in_channels': cfg['aspp']['in_channels'],
            'out_channels': cfg['aspp']['out_channels'],
            'dilations': cfg['aspp']['dilations'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.aspp_net = ASPP(**aspp_cfg)
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
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to aspp
        aspp_out = self.aspp_net(x4)
        # feed to decoder
        preds = self.decoder(aspp_out)
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
            'aspp_net': self.aspp_net, 
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }