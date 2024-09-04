'''
Function:
    Implementation of SemanticFPN
Author:
    Zhenchao Jin
'''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base import FPN, BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''SemanticFPN'''
class SemanticFPN(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(SemanticFPN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build fpn
        self.fpn_neck = FPN(
            in_channels_list=head_cfg['in_channels_list'], out_channels=head_cfg['feats_channels'], upsample_cfg=head_cfg['upsample_cfg'], norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.scale_heads, feature_stride_list = nn.ModuleList(), head_cfg['feature_stride_list']
        for i in range(len(feature_stride_list)):
            head_length = max(1, int(np.log2(feature_stride_list[i]) - np.log2(feature_stride_list[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(nn.Sequential(
                    nn.Conv2d(head_cfg['feats_channels'] if k == 0 else head_cfg['scale_head_channels'], head_cfg['scale_head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(placeholder=head_cfg['scale_head_channels'], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
                if feature_stride_list[i] != feature_stride_list[0]:
                    scale_head.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['scale_head_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to fpn
        fpn_outs = self.fpn_neck(list(backbone_outputs))
        feats = self.scale_heads[0](fpn_outs[0])
        for i in range(1, len(self.cfg['head']['feature_stride_list'])):
            feats = feats + F.interpolate(self.scale_heads[i](fpn_outs[i]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to decoder
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