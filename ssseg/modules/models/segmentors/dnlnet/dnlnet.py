'''
Function:
    Implementation of DNLNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseModel
from .dnlblock import DisentangledNonLocal2d
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''DNLNet'''
class DNLNet(BaseModel):
    def __init__(self, cfg, mode):
        super(DNLNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build disentangled non-local block
        dnl_cfg = cfg['dnl']
        self.conv_before_dnl = nn.Sequential(
            nn.Conv2d(dnl_cfg['in_channels'], dnl_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=dnl_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.dnl_block = DisentangledNonLocal2d(
            in_channels=dnl_cfg['out_channels'],
            reduction=dnl_cfg['reduction'],
            use_scale=dnl_cfg['use_scale'],
            mode=dnl_cfg['mode'],
            temperature=dnl_cfg['temperature'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        self.conv_after_dnl = nn.Sequential(
            nn.Conv2d(dnl_cfg['out_channels'], dnl_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=dnl_cfg['out_channels'], norm_cfg=norm_cfg)),
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
        # feed to disentangled non-local block
        feats = self.conv_before_dnl(backbone_outputs[-1])
        feats = self.dnl_block(feats)
        feats = self.conv_after_dnl(feats)
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
            'conv_before_dnl': self.conv_before_dnl,
            'dnl_block': self.dnl_block,
            'conv_after_dnl': self.conv_after_dnl,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers