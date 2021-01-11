'''
Function:
    Implementation of LRASPPNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel


'''LRASPPNet'''
class LRASPPNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(LRASPPNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build aspp
        aspp_cfg = cfg['aspp']
        self.branch_convs, self.branch_ups = nn.Sequential(), nn.Sequential()
        for idx, branch_channels in enumerate(aspp_cfg['branch_channels_list']):
            self.branch_convs.add_module(
                f'conv{idx}', 
                nn.Conv2d(aspp_cfg['in_channels_list'][idx], branch_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.branch_ups.add_module(
                f'conv{idx}', 
                nn.Sequential(
                    nn.Conv2d(aspp_cfg['out_channels'] + branch_channels, aspp_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(norm_cfg['type'], (aspp_cfg['out_channels'], norm_cfg['opts'])),
                    BuildActivation(act_cfg['type'], **act_cfg['opts']),
                )
            )
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(aspp_cfg['in_channels_list'][-1], aspp_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (aspp_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.image_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=49, stride=(16, 20)),
            nn.Conv2d(aspp_cfg['in_channels_list'][-1], aspp_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (aspp_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation('sigmoid', **{}),
        )
        self.bottleneck = nn.Conv2d(aspp_cfg['out_channels'], aspp_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False)
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to aspp
        feats = self.aspp_conv(outputs[-1]) * F.interpolate(self.image_pool(outputs[-1]), size=outputs[-1].size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats = self.bottleneck(feats)
        for idx in range(len(self.cfg['aspp']['branch_channels_list']) - 1, -1, -1):
            feats = F.interpolate(feats, size=outputs[idx].size()[2:], mode='bilinear', align_corners=self.align_corners)
            feats = torch.cat([feats, self.branch_convs[idx](outputs[idx])], dim=1)
            feats = self.branch_ups[idx](feats)
        # feed to decoder
        preds = self.decoder(feats)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'branch_convs': self.branch_convs,
            'branch_ups': self.branch_ups,
            'aspp_conv': self.aspp_conv,
            'image_pool': self.image_pool,
            'bottleneck': self.bottleneck,
            'decoder': self.decoder,
        }