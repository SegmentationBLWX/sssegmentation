'''
Function:
    Implementation of UPerNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from ..pspnet import PyramidPoolingModule


'''UPerNet'''
class UPerNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(UPerNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': cfg['ppm']['in_channels'],
            'out_channels': cfg['ppm']['out_channels'],
            'pool_scales': cfg['ppm']['pool_scales'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build lateral convs
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy['opts']: act_cfg_copy['opts']['inplace'] = False
        lateral_cfg = cfg['lateral']
        self.lateral_convs = nn.ModuleList()
        for in_channels in lateral_cfg['in_channels_list']:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, lateral_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(norm_cfg['type'], (lateral_cfg['out_channels'], norm_cfg['opts'])),
                    BuildActivation(act_cfg_copy['type'], **act_cfg_copy['opts']),
                )
            )
        # build fpn convs
        fpn_cfg = cfg['fpn']
        self.fpn_convs = nn.ModuleList()
        for in_channels in fpn_cfg['in_channels_list']:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fpn_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(norm_cfg['type'], (fpn_cfg['out_channels'], norm_cfg['opts'])),
                    BuildActivation(act_cfg_copy['type'], **act_cfg_copy['opts']),
                )
            )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
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
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(x4)
        # apply fpn
        inputs = [x1, x2, x3]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        for i in range(len(lateral_outputs) - 1, 0, -1):
            prev_shape = lateral_outputs[i - 1].shape[2:]
            lateral_outputs[i - 1] += F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
        fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
        fpn_outputs.append(lateral_outputs[-1])
        fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
        fpn_out = torch.cat(fpn_outputs, dim=1)
        # feed to decoder
        preds = self.decoder(fpn_out)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(x3)
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_aux': preds_aux}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'ppm_net': self.ppm_net,
            'lateral_convs': self.lateral_convs,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }