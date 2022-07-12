'''
Function:
    Implementation of APCNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseModel
from .acm import AdaptiveContextModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''APCNet'''
class APCNet(BaseModel):
    def __init__(self, cfg, mode):
        super(APCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build acm
        acm_cfg = {
            'in_channels': cfg['acm']['in_channels'],
            'out_channels': cfg['acm']['out_channels'],
            'pool_scale': None,
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.acm_modules = nn.ModuleList()
        for pool_scale in cfg['acm']['pool_scales']:
            acm_cfg['pool_scale'] = pool_scale
            self.acm_modules.append(AdaptiveContextModule(**acm_cfg))
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
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
        # feed to acm
        acm_outs = [backbone_outputs[-1]]
        for acm_module in self.acm_modules:
            acm_outs.append(acm_module(backbone_outputs[-1]))
        feats = torch.cat(acm_outs, dim=1)
        # feed to decoder
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
            'acm_modules': self.acm_modules,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers