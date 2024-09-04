'''
Function:
    Implementation of FCN
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization, DepthwiseSeparableConv2d


'''FCN'''
class FCN(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(FCN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build decoder
        convs = []
        for idx in range(head_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            else:
                conv = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            norm = BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)
            act = BuildActivation(act_cfg)
            convs += [conv, norm, act]
        convs.append(nn.Dropout2d(head_cfg['dropout']))
        if head_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(head_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        seg_logits = self.decoder(backbone_outputs[-1])
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


'''DepthwiseSeparableFCN'''
class DepthwiseSeparableFCN(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(DepthwiseSeparableFCN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build decoder
        convs = []
        for idx in range(head_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = DepthwiseSeparableConv2d(
                    in_channels=head_cfg['in_channels'], out_channels=head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1,
                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
                )
            else:
                conv = DepthwiseSeparableConv2d(
                    in_channels=head_cfg['feats_channels'], out_channels=head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1,
                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
                )
            convs.append(conv)
        convs.append(nn.Dropout2d(head_cfg['dropout']))
        if head_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(head_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        seg_logits = self.decoder(backbone_outputs[-1])
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs