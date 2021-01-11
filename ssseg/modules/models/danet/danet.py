'''
Function:
    Implementation of DANet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .cam import ChannelAttentionModule
from .pam import PositionAttentionModule


'''DANet'''
class DANet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(DANet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build pam and pam decoder
        pam_cfg = cfg['pam']
        self.pam_in_conv = nn.Sequential(
            nn.Conv2d(pam_cfg['in_channels'], pam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (pam_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.pam_net = PositionAttentionModule(pam_cfg['out_channels'], pam_cfg['transform_channels'])
        self.pam_out_conv = nn.Sequential(
            nn.Conv2d(pam_cfg['out_channels'], pam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (pam_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        decoder_cfg = cfg['decoder']['pam']
        self.decoder_pam = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build cam and cam decoder
        cam_cfg = cfg['cam']
        self.cam_in_conv = nn.Sequential(
            nn.Conv2d(cam_cfg['in_channels'], cam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (cam_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.cam_net = ChannelAttentionModule()
        self.cam_out_conv = nn.Sequential(
            nn.Conv2d(cam_cfg['out_channels'], cam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (cam_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        decoder_cfg = cfg['decoder']['cam']
        self.decoder_cam = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build the pam + cam decoder
        decoder_cfg = cfg['decoder']['pamcam']
        self.decoder_pamcam = nn.Sequential(
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
        # feed to pam
        feats_pam = self.pam_in_conv(x4)
        feats_pam = self.pam_net(feats_pam)
        feats_pam = self.pam_out_conv(feats_pam)
        preds_pam = self.decoder_pam(feats_pam)
        # feed to cam
        feats_cam = self.cam_in_conv(x4)
        feats_cam = self.cam_net(feats_cam)
        feats_cam = self.cam_out_conv(feats_cam)
        preds_cam = self.decoder_cam(feats_cam)
        # combine the pam and cam
        feats_sum = feats_pam + feats_cam
        preds_pamcam = self.decoder_pamcam(feats_sum)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds_pam = F.interpolate(preds_pam, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_cam = F.interpolate(preds_cam, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_pamcam = F.interpolate(preds_pamcam, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(x3)
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls_pam': preds_pam, 'loss_cls_cam': preds_cam, 'loss_cls_pamcam': preds_pamcam, 'loss_aux': preds_aux}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds_pamcam
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'pam_in_conv': self.pam_in_conv,
            'pam_net': self.pam_net,
            'pam_out_conv': self.pam_out_conv,
            'decoder_pam': self.decoder_pam,
            'cam_in_conv': self.cam_in_conv,
            'cam_net': self.cam_net,
            'cam_out_conv': self.cam_out_conv,
            'decoder_cam': self.decoder_cam,
            'decoder_pamcam': self.decoder_pamcam,
            'auxiliary_decoder': self.auxiliary_decoder
        }