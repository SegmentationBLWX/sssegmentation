'''
Function:
    Implementation of FCN
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel


'''FCN'''
class FCN(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(FCN, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build decoder
        decoder_cfg = cfg['decoder']
        convs = []
        for idx in range(decoder_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            else:
                conv = nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            norm = BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts']))
            act = BuildActivation(act_cfg['type'], **act_cfg['opts'])
            convs += [conv, norm, act]
        convs.append(nn.Dropout2d(decoder_cfg['dropout']))
        if decoder_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        if auxiliary_cfg is not None:
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
        # feed to decoder
        preds = self.decoder(x4)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            outputs_dict = {'loss_cls': preds}
            if hasattr(self, 'auxiliary_decoder'):
                preds_aux = self.auxiliary_decoder(x3)
                preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                outputs_dict = {'loss_cls': preds, 'loss_aux': preds_aux}
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers