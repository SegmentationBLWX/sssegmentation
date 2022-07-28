'''
Function:
    Implementation of DMNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..base import BaseModel
from .dcm import DynamicConvolutionalModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''DMNet'''
class DMNet(BaseModel):
    def __init__(self, cfg, mode):
        super(DMNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build dcm
        dcm_cfg = cfg['dcm']
        self.dcm_modules = nn.ModuleList()
        for filter_size in dcm_cfg['filter_sizes']:
            self.dcm_modules.append(DynamicConvolutionalModule(
                filter_size=filter_size,
                is_fusion=dcm_cfg['is_fusion'],
                in_channels=dcm_cfg['in_channels'],
                out_channels=dcm_cfg['out_channels'],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ))
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
        # feed to dcm
        dcm_outs = [backbone_outputs[-1]]
        for dcm_module in self.dcm_modules:
            dcm_outs.append(dcm_module(backbone_outputs[-1]))
        feats = torch.cat(dcm_outs, dim=1)
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
            'dcm_modules': self.dcm_modules,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers