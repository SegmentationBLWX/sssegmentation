'''
Function:
    Implementation of EMANet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from .ema import EMAModule
from ..base import BaseModel


'''EMANet'''
class EMANet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(EMANet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build the EMA module
        ema_cfg = cfg['ema']
        self.ema_in_conv = nn.Sequential(
            nn.Conv2d(ema_cfg['in_channels'], ema_cfg['ema_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (ema_cfg['ema_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.ema_mid_conv = nn.Conv2d(ema_cfg['ema_channels'], ema_cfg['ema_channels'], kernel_size=1, stride=1, padding=0)
        for param in self.ema_mid_conv.parameters():
            param.requires_grad = False
        self.ema_module = EMAModule(
            channels=ema_cfg['ema_channels'],
            num_bases=ema_cfg['num_bases'],
            num_stages=ema_cfg['num_stages'],
            momentum=ema_cfg['momentum']
        )
        self.ema_out_conv = nn.Sequential(
            nn.Conv2d(ema_cfg['ema_channels'], ema_cfg['ema_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (ema_cfg['ema_channels'], norm_cfg['opts'])),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ema_cfg['ema_channels'], ema_cfg['ema_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (ema_cfg['ema_channels'], norm_cfg['opts'])),
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
        # feed to EMA module
        feats = self.ema_in_conv(x4)
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = F.relu(recon, inplace=True)
        recon = self.ema_out_conv(recon)
        feats = F.relu(identity + recon, inplace=True)
        feats = self.bottleneck(feats)
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
            'ema_in_conv': self.ema_in_conv,
            'ema_out_conv': self.ema_out_conv,
            'bottleneck': self.bottleneck,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }