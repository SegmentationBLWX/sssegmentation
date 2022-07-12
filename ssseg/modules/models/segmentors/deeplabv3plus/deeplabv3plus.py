'''
Function:
    Implementation of Deeplabv3plus
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from .aspp import DepthwiseSeparableASPP
from ...backbones import BuildActivation, BuildNormalization, DepthwiseSeparableConv2d, constructnormcfg


'''Deeplabv3plus'''
class Deeplabv3Plus(BaseModel):
    def __init__(self, cfg, mode):
        super(Deeplabv3Plus, self).__init__(cfg, mode)
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
        self.aspp_net = DepthwiseSeparableASPP(**aspp_cfg)
        # build shortcut
        shortcut_cfg = cfg['shortcut']
        self.shortcut = nn.Sequential(
            nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=shortcut_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
            DepthwiseSeparableConv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to aspp
        aspp_out = self.aspp_net(backbone_outputs[-1])
        aspp_out = F.interpolate(aspp_out, size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to shortcut
        shortcut_out = self.shortcut(backbone_outputs[0])
        # feed to decoder
        feats = torch.cat([aspp_out, shortcut_out], dim=1)
        predictions = self.decoder(feats)
        # forward according to the mode
        if self.mode == 'TRAIN':
            loss, losses_log_dict = self.forwardtrain(
                predictions=predictions,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
            )
            return loss, losses_log_dict
        return predictions
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net, 
            'aspp_net': self.aspp_net, 
            'shortcut': self.shortcut,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers