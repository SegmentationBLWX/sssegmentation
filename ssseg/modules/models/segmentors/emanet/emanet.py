'''
Function:
    Implementation of EMANet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMAModule
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''EMANet'''
class EMANet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(EMANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build the EMA module
        self.ema_in_conv = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.ema_mid_conv = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0)
        for param in self.ema_mid_conv.parameters():
            param.requires_grad = False
        self.ema_module = EMAModule(
            channels=head_cfg['feats_channels'], num_bases=head_cfg['num_bases'], num_stages=head_cfg['num_stages'], momentum=head_cfg['momentum']
        )
        self.ema_out_conv = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
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
        # feed to EMA module
        feats = self.ema_in_conv(backbone_outputs[-1])
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = F.relu(recon, inplace=True)
        recon = self.ema_out_conv(recon)
        feats = F.relu(identity + recon, inplace=True)
        feats = self.bottleneck(feats)
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