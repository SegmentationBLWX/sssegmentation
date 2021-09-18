'''
Function:
    Implementation of Segformer
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel


'''Segformer'''
class Segformer(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(Segformer, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build decoder
        decoder_cfg = cfg['decoder']
        self.convs = nn.ModuleList()
        for in_channels in decoder_cfg['in_channels_list']:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
                    BuildActivation(act_cfg['type'], **act_cfg['opts']),
                )
            )
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['out_channels'] * len(self.convs), decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        outs = []
        for idx, feats in enumerate([x1, x2, x3, x4]):
            outs.append(
                F.interpolate(self.convs[idx](feats), size=x1.shape[2:], mode='bilinear', align_corners=self.align_corners)
            )
        feats = torch.cat(outs, dim=1)
        preds = self.decoder(feats)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'convs': self.convs,
            'decoder': self.decoder,
        }
        tmp_layers = []
        for key, value in self.backbone_net.zerowdlayers().items():
            tmp_layers.append(value)
        all_layers.update({'backbone_net_zerowd': nn.Sequential(*tmp_layers)})
        tmp_layers = []
        for key, value in self.backbone_net.nonzerowdlayers().items():
            tmp_layers.append(value)
        all_layers.update({'backbone_net_nonzerowd': nn.Sequential(*tmp_layers)})
        return all_layers