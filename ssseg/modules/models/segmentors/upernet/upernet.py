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
from ..base import BaseModel
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''UPerNet'''
class UPerNet(BaseModel):
    def __init__(self, cfg, mode):
        super(UPerNet, self).__init__(cfg, mode)
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
        if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
        lateral_cfg = cfg['lateral']
        self.lateral_convs = nn.ModuleList()
        for in_channels in lateral_cfg['in_channels_list']:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, lateral_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(placeholder=lateral_cfg['out_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg_copy),
            ))
        # build fpn convs
        fpn_cfg = cfg['fpn']
        self.fpn_convs = nn.ModuleList()
        for in_channels in fpn_cfg['in_channels_list']:
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, fpn_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=fpn_cfg['out_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg_copy),
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
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(backbone_outputs[-1])
        # apply fpn
        inputs = backbone_outputs[:-1]
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
        predictions = self.decoder(fpn_out)
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
        if self.cfg['backbone']['series'] in ['swin', 'twins']:
            all_layers = {
                'ppm_net': self.ppm_net,
                'lateral_convs': self.lateral_convs,
                'decoder': self.decoder,
            }
            if hasattr(self, 'auxiliary_decoder'):
                all_layers['auxiliary_decoder'] = self.auxiliary_decoder
            tmp_layers = []
            for key, value in self.backbone_net.zerowdlayers().items():
                tmp_layers.append(value)
            all_layers.update({'backbone_net_zerowd': nn.Sequential(*tmp_layers)})
            tmp_layers = []
            for key, value in self.backbone_net.nonzerowdlayers().items():
                tmp_layers.append(value)
            all_layers.update({'backbone_net_nonzerowd': nn.Sequential(*tmp_layers)})
            return all_layers
        else:
            all_layers = {
                'backbone_net': self.backbone_net,
                'ppm_net': self.ppm_net,
                'lateral_convs': self.lateral_convs,
                'decoder': self.decoder,
            }
            if hasattr(self, 'auxiliary_decoder'):
                all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers