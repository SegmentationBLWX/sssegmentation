'''
Function:
    Implementation of ENCNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from .contextencoding import ContextEncoding
from ...backbones import BuildActivation, BuildNormalization


'''ENCNet'''
class ENCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ENCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build encoding
        # --base structurs
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), 
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg),
        )
        self.enc_module = ContextEncoding(in_channels=head_cfg['feats_channels'], num_codes=head_cfg['num_codes'], norm_cfg=norm_cfg, act_cfg=act_cfg)
        # --extra structures
        extra_cfg = head_cfg['extra']
        if extra_cfg['add_lateral']:
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['in_channels_list'][:-1]:
                self.lateral_convs.append(
                    nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0), 
                    BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg),
                )
            self.fusion = nn.Sequential(
                nn.Conv2d(len(head_cfg['in_channels_list']) * head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1), 
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg),
            )
        if extra_cfg['use_se_loss']:
            self.se_layer = nn.Linear(head_cfg['feats_channels'], cfg['num_classes'])
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
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
        # feed to context encoding
        feats = self.bottleneck(backbone_outputs[-1])
        if hasattr(self, 'lateral_convs'):
            lateral_outs = [F.interpolate(lateral_conv(backbone_outputs[idx]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners) for idx, lateral_conv in enumerate(self.lateral_convs)]
            feats = self.fusion(torch.cat([feats, *lateral_outs], dim=1))
        encode_feats, feats = self.enc_module(feats)
        if hasattr(self, 'se_layer'):
            seg_logits_se = self.se_layer(encode_feats)
        # feed to decoder
        seg_logits = self.decoder(feats)
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False,
            )
            map_preds_to_tgts_dict, targets = None, data_meta.gettargets()
            if hasattr(self, 'se_layer'):
                predictions.update({'loss_se': seg_logits_se})
                targets['seg_targets_onehot'] = self.onehot(targets['seg_targets'], self.cfg['num_classes'])
                map_preds_to_tgts_dict = {'loss_aux': 'seg_targets', 'loss_se': 'seg_targets_onehot', 'loss_cls': 'seg_targets'}
            loss, losses_log_dict = self.calculatelosses(
                predictions=predictions, targets=targets, losses_cfg=self.cfg['losses'], map_preds_to_tgts_dict=map_preds_to_tgts_dict,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
    '''convert to onehot labels'''
    def onehot(self, labels, num_classes):
        batch_size = labels.size(0)
        labels_onehot = labels.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = labels[i].float().histc(bins=num_classes, min=0, max=num_classes-1)
            labels_onehot[i] = hist > 0
        return labels_onehot