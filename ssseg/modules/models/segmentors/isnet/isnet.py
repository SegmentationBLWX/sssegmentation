'''
Function:
    Implementation of ISNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from .imagelevel import ImageLevelContext
from ....utils import SSSegOutputStructure
from .semanticlevel import SemanticLevelContext
from ...backbones import BuildActivation, BuildNormalization


'''ISNet'''
class ISNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ISNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build image-level context module
        ilc_cfg = {
            'feats_channels': head_cfg['feats_channels'], 'transform_channels': head_cfg['transform_channels'], 'concat_input': head_cfg['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg), 'align_corners': align_corners,
        }
        self.ilc_net = ImageLevelContext(**ilc_cfg)
        # build semantic-level context module
        slc_cfg = {
            'feats_channels': head_cfg['feats_channels'], 'transform_channels': head_cfg['transform_channels'], 'concat_input': head_cfg['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)
        # build decoder
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        if head_cfg['shortcut']['is_on']:
            self.shortcut = nn.Sequential(
                nn.Conv2d(head_cfg['shortcut']['in_channels'], head_cfg['shortcut']['feats_channels'], kernel_size=1, stride=1, padding=0),
                BuildNormalization(placeholder=head_cfg['shortcut']['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            self.decoder_stage2 = nn.Sequential(
                nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut']['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Dropout2d(head_cfg['dropout']),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        else:
            self.decoder_stage2 = nn.Sequential(
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
        # feed to bottleneck
        feats = self.bottleneck(backbone_outputs[-1])
        # feed to image-level context module
        feats_il = self.ilc_net(feats)
        # feed to decoder stage1
        preds_stage1 = self.decoder_stage1(feats)
        # feed to semantic-level context module
        preds = preds_stage1
        if preds_stage1.size()[2:] != feats.size()[2:]:
            preds = F.interpolate(preds_stage1, size=feats.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_sl = self.slc_net(feats, preds, feats_il)
        # feed to decoder stage2
        if hasattr(self, 'shortcut'):
            shortcut_out = self.shortcut(backbone_outputs[0])
            feats_sl = F.interpolate(feats_sl, size=shortcut_out.shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
        preds_stage2 = self.decoder_stage2(feats_sl)
        # return according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(
                seg_logits=preds_stage2, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False,
            )
            preds_stage2 = predictions.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            loss, losses_log_dict = self.calculatelosses(
                predictions=predictions, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses']
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs