'''
Function:
    Implementation of OCRNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule


'''OCRNet'''
class OCRNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(OCRNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(auxiliary_cfg['dropout']),
            nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build bottleneck
        bottleneck_cfg = cfg['bottleneck']
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_cfg['in_channels'], bottleneck_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (bottleneck_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': cfg['spatialgather']['scale']
        }
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        # build object context block
        ocb_cfg = cfg['objectcontext']
        self.object_context_block = ObjectContextBlock(
            in_channels=ocb_cfg['in_channels'], 
            transform_channels=ocb_cfg['transform_channels'], 
            scale=ocb_cfg['scale'],
            align_corners=align_corners,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
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
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to bottleneck
        feats = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats, preds_aux)
        feats = self.object_context_block(feats, context)
        # feed to decoder
        preds = self.decoder(feats)
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
            'bottleneck': self.bottleneck,
            'spatial_gather_module': self.spatial_gather_module,
            'object_context_block': self.object_context_block,
            'decoder': self.decoder
        }