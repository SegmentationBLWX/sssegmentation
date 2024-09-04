'''
Function:
    Implementation of DMNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from .dcm import DynamicConvolutionalModule
from ...backbones import BuildActivation, BuildNormalization


'''DMNet'''
class DMNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(DMNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build dcm
        self.dcm_modules = nn.ModuleList()
        for filter_size in head_cfg['filter_sizes']:
            self.dcm_modules.append(DynamicConvolutionalModule(
                filter_size=filter_size, is_fusion=head_cfg['is_fusion'], in_channels=head_cfg['in_channels'],
                out_channels=head_cfg['feats_channels'], norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['filter_sizes']) + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
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
        # feed to dcm
        dcm_outs = [backbone_outputs[-1]]
        for dcm_module in self.dcm_modules:
            dcm_outs.append(dcm_module(backbone_outputs[-1]))
        feats = torch.cat(dcm_outs, dim=1)
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