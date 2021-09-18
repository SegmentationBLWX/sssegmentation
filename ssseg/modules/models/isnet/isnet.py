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
from ...backbones import *
from ..base import BaseModel
from .imagelevel import ImageLevelContext
from .semanticlevel import SemanticLevelContext


'''ISNet'''
class ISNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(ISNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build bottleneck
        bottleneck_cfg = cfg['bottleneck']
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_cfg['in_channels'], bottleneck_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (bottleneck_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build image-level context module
        ilc_cfg = {
            'feats_channels': cfg['imagelevel']['feats_channels'],
            'transform_channels': cfg['imagelevel']['transform_channels'],
            'concat_input': cfg['imagelevel']['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
            'align_corners': align_corners,
        }
        self.ilc_net = ImageLevelContext(**ilc_cfg)
        # build semantic-level context module
        slc_cfg = {
            'feats_channels': cfg['semanticlevel']['feats_channels'],
            'transform_channels': cfg['semanticlevel']['transform_channels'],
            'concat_input': cfg['semanticlevel']['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)
        # build decoder
        decoder_cfg = cfg['decoder']['stage1']
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        decoder_cfg, shortcut_cfg = cfg['decoder']['stage2'], cfg['shortcut']
        if shortcut_cfg['is_on']:
            self.shortcut = nn.Sequential(
                nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0),
                BuildNormalization(norm_cfg['type'], (shortcut_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            )
            self.decoder_stage2 = nn.Sequential(
                nn.Conv2d(decoder_cfg['out_channels']+shortcut_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Dropout2d(decoder_cfg['dropout']),
                nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        else:
            self.decoder_stage2 = nn.Sequential(
                nn.Dropout2d(decoder_cfg['dropout']),
                nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        if auxiliary_cfg is not None:
            self.auxiliary_decoder = nn.Sequential(
                nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Dropout2d(auxiliary_cfg['dropout']),
                nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to bottleneck
        feats = self.bottleneck(x4)
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
            shortcut_out = self.shortcut(x1)
            feats_sl = F.interpolate(feats_sl, size=shortcut_out.shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
        preds_stage2 = self.decoder_stage2(feats_sl)
        # return according to the mode
        if self.mode == 'TRAIN':
            preds_stage1 = F.interpolate(preds_stage1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_stage2 = F.interpolate(preds_stage2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            outputs_dict = {'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2}
            if hasattr(self, 'auxiliary_decoder'):
                preds_aux = self.auxiliary_decoder(x3)
                preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                outputs_dict = {'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2, 'loss_aux': preds_aux}
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds_stage2
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net,
            'bottleneck': self.bottleneck,
            'ilc_net': self.ilc_net,
            'slc_net': self.slc_net,
            'decoder_stage1': self.decoder_stage1,
            'decoder_stage2': self.decoder_stage2,
        }
        if hasattr(self, 'shortcut'):
            all_layers.update({
                'shortcut': self.shortcut
            })
        if hasattr(self, 'auxiliary_decoder'):
            all_layers.update({
                'auxiliary_decoder': self.auxiliary_decoder
            })
        return all_layers