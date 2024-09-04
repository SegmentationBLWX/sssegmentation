'''
Function:
    Implementation of DANet
Author:
    Zhenchao Jin
'''
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from .cam import ChannelAttentionModule
from .pam import PositionAttentionModule
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''DANet'''
class DANet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(DANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build pam and pam decoder
        self.pam_in_conv = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.pam_net = PositionAttentionModule(head_cfg['feats_channels'], head_cfg['transform_channels'])
        self.pam_out_conv = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.decoder_pam = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build cam and cam decoder
        self.cam_in_conv = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.cam_net = ChannelAttentionModule()
        self.cam_out_conv = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.decoder_cam = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build the pam + cam decoder
        self.decoder_pamcam = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
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
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(
                seg_logits=preds_pamcam, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False,
            )
            preds_pamcam = predictions.pop('loss_cls')
            preds_pam = self.decoder_pam(feats_pam)
            preds_pam = F.interpolate(preds_pam, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_cam = self.decoder_cam(feats_cam)
            preds_cam = F.interpolate(preds_cam, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_pam': preds_pam, 'loss_cls_cam': preds_cam, 'loss_cls_pamcam': preds_pamcam})
            loss, losses_log_dict = self.calculatelosses(
                predictions=predictions, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses']
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_pamcam)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_pamcam)
        return ssseg_outputs