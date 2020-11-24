'''
Function:
    Implementation of CE2P
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .epm import EdgePerceivingModule
from ..pspnet import PyramidPoolingModule


'''ce2p'''
class CE2P(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(CE2P, self).__init__(cfg, **kwargs)
        align_corners, normlayer_opts, activation_opts = self.align_corners, self.normlayer_opts, self.activation_opts
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': cfg['ppm']['in_channels'],
            'out_channels': cfg['ppm']['out_channels'],
            'bin_sizes': cfg['ppm']['bin_sizes'],
            'align_corners': align_corners,
            'normlayer_opts': copy.deepcopy(normlayer_opts),
            'activation_opts': copy.deepcopy(activation_opts),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        lateral_ppm_cfg = cfg['lateral_ppm']
        self.lateral_ppm_layer = nn.Sequential(
            nn.Conv2d(lateral_ppm_cfg['in_channels'], lateral_ppm_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(normlayer_opts['type'], (lateral_ppm_cfg['out_channels'], normlayer_opts['opts'])),
            BuildActivation(activation_opts['type'], **activation_opts['opts'])
        )
        # build edge perceiving module
        epm_cfg = {
            'in_channels_list': cfg['epm']['in_channels_list'],
            'hidden_channels': cfg['epm']['hidden_channels'],
            'out_channels': cfg['epm']['out_channels'],
            'align_corners': align_corners,
            'normlayer_opts': copy.deepcopy(normlayer_opts),
            'activation_opts': copy.deepcopy(activation_opts),
        }
        self.edge_net = EdgePerceivingModule(**epm_cfg)
        # build shortcut
        shortcut_cfg = cfg['shortcut']
        self.shortcut = nn.Sequential(
            nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(normlayer_opts['type'], (shortcut_cfg['out_channels'], normlayer_opts['opts'])),
            BuildActivation(activation_opts['type'], **activation_opts['opts'])
        )
        # build decoder stage1
        decoder_cfg = cfg['decoder']['stage1']
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(normlayer_opts['type'], (decoder_cfg['out_channels'], normlayer_opts['opts'])),
            BuildActivation(activation_opts['type'], **activation_opts['opts']),
            nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(normlayer_opts['type'], (decoder_cfg['out_channels'], normlayer_opts['opts'])),
            BuildActivation(activation_opts['type'], **activation_opts['opts']),
            nn.Dropout2d(decoder_cfg['dropout']), 
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build decoder stage1
        decoder_cfg = cfg['decoder']['stage2']
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalizationLayer(normlayer_opts['type'], (decoder_cfg['out_channels'], normlayer_opts['opts'])),
            BuildActivation(activation_opts['type'], **activation_opts['opts']),
            nn.Dropout2d(decoder_cfg['dropout']), 
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_normlayer', False): self.freezenormlayer()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.backbone_net(x)
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(x4)
        ppm_out = self.lateral_ppm_layer(ppm_out)
        ppm_out = F.interpolate(ppm_out, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=self.align_corners)
        # feed to edge perceiving module
        edge, edge_feats = self.edge_net((x1, x2, x3))
        edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        # feed to shortcut
        shortcut_out = self.shortcut(x1)
        # feed to decoder stage1
        features = torch.cat([ppm_out, shortcut_out], dim=1)
        features = self.decoder_stage1[:-1](features)
        preds_stage1 = self.decoder_stage1[-1](features)
        preds_stage1 = F.interpolate(preds_stage1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        # feed to decoder stage2
        features = torch.cat([features, edge_feats], dim=1)
        preds_stage2 = self.decoder_stage2(features)
        # return according to the mode
        if self.mode == 'TRAIN':
            preds_stage2 = F.interpolate(preds_stage2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2, 'loss_edge': edge}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds_stage2
    '''return all layers'''
    def alllayers(self):
        return {
                'backbone_net': self.backbone_net,
                'ppm_net': self.ppm_net,
                'lateral_ppm_layer': self.lateral_ppm_layer,
                'edge_net': self.edge_net,
                'shortcut': self.shortcut,
                'decoder_stage1': self.decoder_stage1,
                'decoder_stage2': self.decoder_stage2,
            }