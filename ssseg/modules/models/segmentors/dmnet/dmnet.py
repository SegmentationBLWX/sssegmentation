'''
Function:
    Implementation of DMNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..base import BaseSegmentor
from .dcm import DynamicConvolutionalModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''DMNet'''
class DMNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(DMNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build dcm
        self.dcm_modules = nn.ModuleList()
        for filter_size in head_cfg['filter_sizes']:
            self.dcm_modules.append(DynamicConvolutionalModule(
                filter_size=filter_size,
                is_fusion=head_cfg['is_fusion'],
                in_channels=head_cfg['in_channels'],
                out_channels=head_cfg['feats_channels'],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ))
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['filter_sizes']) + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'dcm_modules', 'decoder', 'auxiliary_decoder']
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