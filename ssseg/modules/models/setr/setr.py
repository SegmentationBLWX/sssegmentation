'''
Function:
    Implementation of SETR
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mla import MLANeck
from ...backbones import *
from ..base import BaseModel


'''Naive upsampling head and Progressive upsampling head of SETR'''
class SETRUP(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(SETRUP, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build norm layer
        self.norm_layers = nn.ModuleList()
        for in_channels in cfg['normlayer']['in_channels_list']:
            norm_layer = BuildNormalization(cfg['normlayer']['type'], (in_channels, cfg['normlayer']['opts']))
            self.norm_layers.append(norm_layer)
        # build decoder
        self.decoder = self.builddecoder(cfg['decoder'])
        # build auxiliary decoder
        auxiliary_cfg_list = cfg['auxiliary']
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to norm layer
        x1 = self.norm(x1, self.norm_layers[0])
        x2 = self.norm(x2, self.norm_layers[1])
        x3 = self.norm(x3, self.norm_layers[2])
        x4 = self.norm(x4, self.norm_layers[3])
        # feed to decoder
        preds = self.decoder(x4)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds_aux1 = self.auxiliary_decoders[0](x1)
            preds_aux2 = self.auxiliary_decoders[1](x2)
            preds_aux3 = self.auxiliary_decoders[2](x3)
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux1 = F.interpolate(preds_aux1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux2 = F.interpolate(preds_aux2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux3 = F.interpolate(preds_aux3, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_aux1': preds_aux1, 'loss_aux2': preds_aux2, 'loss_aux3': preds_aux3}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''norm layer'''
    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    '''build decoder'''
    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])))
            layers.append(BuildActivation(act_cfg['type'], **act_cfg['opts']))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'norm_layers': self.norm_layers,
            'decoder': self.decoder,
            'auxiliary_decoders': self.auxiliary_decoders
        }


'''Multi level feature aggretation head of SETR'''
class SETRMLA(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(SETRMLA, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build mla neck
        norm_layers = nn.ModuleList()
        for in_channels in cfg['normlayer']['in_channels_list']:
            norm_layer = BuildNormalization(cfg['normlayer']['type'], (in_channels, cfg['normlayer']['opts']))
            norm_layers.append(norm_layer)
        mla_cfg = cfg['mla']
        self.mla_neck = MLANeck(
            in_channels_list=mla_cfg['in_channels_list'], 
            out_channels=mla_cfg['out_channels'], 
            norm_layers=norm_layers, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        # build upsample convs and decoder
        decoder_cfg = cfg['decoder']
        assert decoder_cfg['mla_channels'] * len(decoder_cfg['in_channels_list']) == decoder_cfg['out_channels']
        self.up_convs = nn.ModuleList()
        for i in range(len(decoder_cfg['in_channels_list'])):
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(decoder_cfg['in_channels_list'][i], decoder_cfg['mla_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(norm_cfg['type'], (decoder_cfg['mla_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Conv2d(decoder_cfg['mla_channels'], decoder_cfg['mla_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(norm_cfg['type'], (decoder_cfg['mla_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners)
            ))
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # build auxiliary decoder
        auxiliary_cfg_list = cfg['auxiliary']
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to mla neck
        feats_list = self.mla_neck([x1, x2, x3, x4])
        # feed to decoder
        outputs = []
        assert len(feats_list) == len(self.up_convs)
        for feats, up_conv in zip(feats_list, self.up_convs):
            outputs.append(up_conv(feats))
        outputs = torch.cat(outputs, dim=1)
        preds = self.decoder(outputs)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds_aux1 = self.auxiliary_decoders[0](feats_list[0])
            preds_aux2 = self.auxiliary_decoders[1](feats_list[1])
            preds_aux3 = self.auxiliary_decoders[2](feats_list[2])
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux1 = F.interpolate(preds_aux1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux2 = F.interpolate(preds_aux2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux3 = F.interpolate(preds_aux3, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_aux1': preds_aux1, 'loss_aux2': preds_aux2, 'loss_aux3': preds_aux3}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''build decoder'''
    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])))
            layers.append(BuildActivation(act_cfg['type'], **act_cfg['opts']))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'mla_neck': self.mla_neck,
            'up_convs': self.up_convs,
            'decoder': self.decoder,
            'auxiliary_decoders': self.auxiliary_decoders
        }