'''
Function:
    Implementation of DANet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from .cam import ChannelAttentionModule
from .pam import PositionAttentionModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''DANet'''
class DANet(BaseModel):
    def __init__(self, cfg, mode):
        super(DANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build pam and pam decoder
        pam_cfg = cfg['pam']
        self.pam_in_conv = nn.Sequential(
            nn.Conv2d(pam_cfg['in_channels'], pam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=pam_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.pam_net = PositionAttentionModule(pam_cfg['out_channels'], pam_cfg['transform_channels'])
        self.pam_out_conv = nn.Sequential(
            nn.Conv2d(pam_cfg['out_channels'], pam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=pam_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
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
            BuildNormalization(constructnormcfg(placeholder=cam_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.cam_net = ChannelAttentionModule()
        self.cam_out_conv = nn.Sequential(
            nn.Conv2d(cam_cfg['out_channels'], cam_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=cam_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
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
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to pam
        feats_pam = self.pam_in_conv(backbone_outputs[-1])
        feats_pam = self.pam_net(feats_pam)
        feats_pam = self.pam_out_conv(feats_pam)
        # feed to cam
        feats_cam = self.cam_in_conv(backbone_outputs[-1])
        feats_cam = self.cam_net(feats_cam)
        feats_cam = self.cam_out_conv(feats_cam)
        # combine the pam and cam
        feats_sum = feats_pam + feats_cam
        preds_pamcam = self.decoder_pamcam(feats_sum)
        # forward according to the mode
        if self.mode == 'TRAIN':
            outputs_dict = self.forwardtrain(
                predictions=preds_pamcam,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
                compute_loss=False,
            )
            preds_pamcam = outputs_dict.pop('loss_cls')
            preds_pam = self.decoder_pam(feats_pam)
            preds_pam = F.interpolate(preds_pam, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_cam = self.decoder_cam(feats_cam)
            preds_cam = F.interpolate(preds_cam, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict.update({'loss_cls_pam': preds_pam, 'loss_cls_cam': preds_cam, 'loss_cls_pamcam': preds_pamcam})
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds_pamcam
    '''return all layers'''
    def alllayers(self):
        all_layers = {
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
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers