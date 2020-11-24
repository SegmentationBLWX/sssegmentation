'''
Function:
    define the ocrnet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .spatialocr import SpatialOCRModule
from .spatialgather import SpatialGatherModule


'''OCRNet'''
class OCRNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(OCRNet, self).__init__(cfg, **kwargs)
        align_corners, normlayer_opts, activation_opts = self.align_corners, self.normlayer_opts, self.activation_opts
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalizationLayer(normlayer_opts['type'], (auxiliary_cfg['out_channels'], normlayer_opts['opts'])),
            BuildActivation(activation_opts['type'], **activation_opts['opts']),
            nn.Dropout2d(auxiliary_cfg['dropout']),
            nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': cfg['spatialgather']['scale']
        }
        self.spatial_gather_net = SpatialGatherModule(**spatialgather_cfg)
        # build spatial ocr module
        spatialocr_cfg = {
            'in_channels': cfg['spatialocr']['in_channels'],
            'key_channels': cfg['spatialocr']['key_channels'],
            'out_channels': cfg['spatialocr']['out_channels'],
            'align_corners': align_corners,
            'normlayer_opts': normlayer_opts,
            'activation_opts': activation_opts,
        }
        self.spatial_ocr_net = SpatialOCRModule(**spatialocr_cfg)
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_normlayer', False): self.freezenormlayer()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.backbone_net(x)
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to ocr module
        feats = F.interpolate(x4, size=preds_aux.size()[2:], mode='bilinear', align_corners=self.align_corners)
        context = self.spatial_gather_net(feats, preds_aux)
        features = self.spatial_ocr_net(feats, context)
        # feed to decoder
        preds = self.decoder(features)
        # return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
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
                'auxiliary_decoder': self.auxiliary_decoder,
                'spatial_gather_net': self.spatial_gather_net,
                'spatial_ocr_net': self.spatial_ocr_net,
                'decoder': self.decoder
            }