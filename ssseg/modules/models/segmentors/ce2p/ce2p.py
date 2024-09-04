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
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''CE2P'''
class CE2P(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(CE2P, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'pool_scales': head_cfg['pool_scales'],
            'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build edge perceiving module
        epm_cfg = {
            'in_channels_list': head_cfg['in_channels_list'][:-1], 'hidden_channels': head_cfg['epm_hidden_channels'], 'out_channels': head_cfg['epm_out_channels'],
            'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
        }
        self.edge_net = EdgePerceivingModule(**epm_cfg)
        # build shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels_list'][0], head_cfg['shortcut_feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['shortcut_feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build decoder stage1
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut_feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout_stage1']), 
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build decoder stage1
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] + head_cfg['epm_hidden_channels'] * (len(head_cfg['in_channels_list']) - 1), head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout_stage2']), 
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
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
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            edge = F.interpolate(edge, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage1 = self.decoder_stage1[-1](feats_stage1)
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage2 = F.interpolate(preds_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
            edge_targets, losses_cfg = data_meta.gettargets()['edge_targets'], copy.deepcopy(self.cfg['losses'])
            num_neg_edge, num_pos_edge = torch.sum(edge_targets == 0, dtype=torch.float), torch.sum(edge_targets == 1, dtype=torch.float)
            weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
            cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(edge_targets)
            for loss_name in list(losses_cfg.keys()):
                if 'edge' in loss_name:
                    if isinstance(losses_cfg[loss_name], list):
                        for loss_idx in range(len(losses_cfg[loss_name])):
                            losses_cfg[loss_name][loss_idx]['weight'] = cls_weight_edge
                    else:
                        assert isinstance(losses_cfg[loss_name], dict)
                        losses_cfg[loss_name]['weight'] = cls_weight_edge
            loss, losses_log_dict = self.calculatelosses(
                predictions={'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2, 'loss_edge': edge}, targets=data_meta.gettargets(), losses_cfg=losses_cfg,
                map_preds_to_tgts_dict={'loss_cls_stage1': 'seg_targets', 'loss_cls_stage2': 'seg_targets', 'loss_edge': 'edge_targets'},
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs