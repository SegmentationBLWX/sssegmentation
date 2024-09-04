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
from ..base import BaseSegmentor
from .aspp import DepthwiseSeparableASPP
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization, DepthwiseSeparableConv2d


'''Deeplabv3plus'''
class Deeplabv3Plus(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(Deeplabv3Plus, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build aspp net
        aspp_cfg = {
            'in_channels': head_cfg['in_channels'][1], 'out_channels': head_cfg['feats_channels'], 'dilations': head_cfg['dilations'],
            'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
        }
        self.aspp_net = DepthwiseSeparableASPP(**aspp_cfg)
        # build shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'][0], head_cfg['shortcut_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['shortcut_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(head_cfg['feats_channels'] + head_cfg['shortcut_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
            DepthwiseSeparableConv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
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
        # feed to aspp
        aspp_out = self.aspp_net(backbone_outputs[-1])
        aspp_out = F.interpolate(aspp_out, size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to shortcut
        shortcut_out = self.shortcut(backbone_outputs[0])
        # feed to decoder
        feats = torch.cat([aspp_out, shortcut_out], dim=1)
        seg_logits = self.decoder(feats)
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs