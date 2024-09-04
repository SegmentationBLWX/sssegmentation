'''
Function:
    Implementation of LRASPPNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''LRASPPNet'''
class LRASPPNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(LRASPPNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build aspp
        self.branch_convs, self.branch_ups = nn.Sequential(), nn.Sequential()
        for idx, branch_channels in enumerate(head_cfg['branch_channels_list']):
            self.branch_convs.add_module(
                f'conv{idx}', nn.Conv2d(head_cfg['in_channels_list'][idx], branch_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.branch_ups.add_module(f'conv{idx}', nn.Sequential(
                nn.Conv2d(head_cfg['feats_channels'] + branch_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            ))
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.image_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=49, stride=(16, 20)),
            nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            nn.Sigmoid(),
        )
        self.bottleneck = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False)
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to aspp
        feats = self.aspp_conv(backbone_outputs[-1]) * F.interpolate(self.image_pool(backbone_outputs[-1]), size=backbone_outputs[-1].size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats = self.bottleneck(feats)
        for idx in range(len(self.cfg['head']['branch_channels_list']) - 1, -1, -1):
            feats = F.interpolate(feats, size=backbone_outputs[idx].size()[2:], mode='bilinear', align_corners=self.align_corners)
            feats = torch.cat([feats, self.branch_convs[idx](backbone_outputs[idx])], dim=1)
            feats = self.branch_ups[idx](feats)
        # feed to decoder
        seg_logits = self.decoder(feats)
        # return according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs