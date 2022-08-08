'''
Function:
    Implementation of CE2P
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from .epm import EdgePerceivingModule
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''CE2P'''
class CE2P(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(CE2P, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': head_cfg['in_channels_list'][-1],
            'out_channels': head_cfg['feats_channels'],
            'pool_scales': head_cfg['pool_scales'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build edge perceiving module
        epm_cfg = {
            'in_channels_list': head_cfg['in_channels_list'][:-1],
            'hidden_channels': head_cfg['epm_hidden_channels'],
            'out_channels': head_cfg['epm_out_channels'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.edge_net = EdgePerceivingModule(**epm_cfg)
        # build shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels_list'][0], head_cfg['shortcut_feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['shortcut_feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder stage1
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut_feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout_stage1']), 
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build decoder stage1
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] + head_cfg['epm_hidden_channels'] * (len(head_cfg['in_channels_list']) - 1), head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout_stage2']), 
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'ppm_net', 'edge_net', 'shortcut', 'decoder_stage1', 'decoder_stage2']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(backbone_outputs[-1])
        ppm_out = F.interpolate(ppm_out, size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to edge perceiving module
        edge, edge_feats = self.edge_net(backbone_outputs[:-1])
        # feed to shortcut
        shortcut_out = self.shortcut(backbone_outputs[0])
        # feed to decoder stage1
        feats_stage1 = torch.cat([ppm_out, shortcut_out], dim=1)
        feats_stage1 = self.decoder_stage1[:-1](feats_stage1)
        # feed to decoder stage2
        feats_stage2 = torch.cat([feats_stage1, edge_feats], dim=1)
        preds_stage2 = self.decoder_stage2(feats_stage2)
        # forward according to the mode
        if self.mode == 'TRAIN':
            edge = F.interpolate(edge, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage1 = self.decoder_stage1[-1](feats_stage1)
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage2 = F.interpolate(preds_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2, 'loss_edge': edge}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds_stage2