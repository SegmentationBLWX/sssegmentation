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


'''CE2P'''
class CE2P(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(CE2P, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': cfg['ppm']['in_channels'],
            'out_channels': cfg['ppm']['out_channels'],
            'pool_scales': cfg['ppm']['pool_scales'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build edge perceiving module
        epm_cfg = {
            'in_channels_list': cfg['epm']['in_channels_list'],
            'hidden_channels': cfg['epm']['hidden_channels'],
            'out_channels': cfg['epm']['out_channels'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.edge_net = EdgePerceivingModule(**epm_cfg)
        # build shortcut
        shortcut_cfg = cfg['shortcut']
        self.shortcut = nn.Sequential(
            nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (shortcut_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        # build decoder stage1
        decoder_cfg = cfg['decoder']['stage1']
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']), 
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build decoder stage1
        decoder_cfg = cfg['decoder']['stage2']
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']), 
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(x4)
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
            'edge_net': self.edge_net,
            'shortcut': self.shortcut,
            'decoder_stage1': self.decoder_stage1,
            'decoder_stage2': self.decoder_stage2,
        }