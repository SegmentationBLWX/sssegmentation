'''
Function:
    Implementation of GCNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseSegmentor
from .contextblock import ContextBlock
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''GCNet'''
class GCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(GCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build context block
        self.conv_before_cb = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.contextblock_net = ContextBlock(
            in_channels=head_cfg['feats_channels'], ratio=head_cfg['ratio'], pooling_type=head_cfg['pooling_type'], fusion_types=head_cfg['fusion_types'],
            norm_cfg=head_cfg.get('norm_cfg', copy.deepcopy(norm_cfg)), act_cfg=head_cfg.get('act_cfg', copy.deepcopy(act_cfg)),
        )
        self.conv_after_cb = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'] + head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
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
        # feed to context block
        feats = self.conv_before_cb(backbone_outputs[-1])
        feats = self.contextblock_net(feats)
        feats = self.conv_after_cb(feats)
        # feed to decoder
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
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