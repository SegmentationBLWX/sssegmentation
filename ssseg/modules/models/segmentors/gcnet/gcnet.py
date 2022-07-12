'''
Function:
    Implementation of GCNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseModel
from .contextblock import ContextBlock
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''GCNet'''
class GCNet(BaseModel):
    def __init__(self, cfg, mode):
        super(GCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build context block
        cb_cfg = cfg['contextblock']
        self.conv_before_cb = nn.Sequential(
            nn.Conv2d(cb_cfg['in_channels'], cb_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=cb_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.contextblock_net = ContextBlock(
            in_channels=cb_cfg['out_channels'],
            ratio=cb_cfg['ratio'],
            pooling_type=cb_cfg['pooling_type'],
            fusion_types=cb_cfg['fusion_types'],
            norm_cfg=cb_cfg.get('norm_cfg', copy.deepcopy(norm_cfg)),
            act_cfg=cb_cfg.get('act_cfg', copy.deepcopy(act_cfg)),
        )
        self.conv_after_cb = nn.Sequential(
            nn.Conv2d(cb_cfg['out_channels'], cb_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=cb_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to context block
        feats = self.conv_before_cb(backbone_outputs[-1])
        feats = self.contextblock_net(feats)
        feats = self.conv_after_cb(feats)
        # feed to decoder
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
        predictions = self.decoder(feats)
        # forward according to the mode
        if self.mode == 'TRAIN':
            loss, losses_log_dict = self.forwardtrain(
                predictions=predictions,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
            )
            return loss, losses_log_dict
        return predictions
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net,
            'conv_before_cb': self.conv_before_cb,
            'contextblock_net': self.contextblock_net,
            'conv_after_cb': self.conv_after_cb,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers