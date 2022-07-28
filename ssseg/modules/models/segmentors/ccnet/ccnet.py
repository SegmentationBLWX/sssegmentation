'''
Function:
    Implementation of CCNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..base import BaseModel
from mmcv.ops import CrissCrossAttention
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''CCNet'''
class CCNet(BaseModel):
    def __init__(self, cfg, mode):
        super(CCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build criss-cross attention
        cca_cfg = cfg['cca']
        self.conv_before_cca = nn.Sequential(
            nn.Conv2d(cca_cfg['in_channels'], cca_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=cca_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.cca = CrissCrossAttention(cca_cfg['out_channels'])
        self.conv_after_cca = nn.Sequential(
            nn.Conv2d(cca_cfg['out_channels'], cca_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=cca_cfg['out_channels'], norm_cfg=norm_cfg)),
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
        # feed to cca
        feats = self.conv_before_cca(backbone_outputs[-1])
        for _ in range(self.cfg['cca']['num_recurrence']):
            feats = self.cca(feats)
        feats = self.conv_after_cca(feats)
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
            'conv_before_cca': self.conv_before_cca,
            'cca': self.cca,
            'conv_after_cca': self.conv_before_cca,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers