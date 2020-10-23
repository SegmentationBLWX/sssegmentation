'''
Function:
    Implementation of Deeplabv3+
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...losses import *
from ...backbones import *
from .aspp import DepthwiseSeparableASPP


'''deeplabv3 plus'''
class Deeplabv3Plus(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Deeplabv3Plus, self).__init__()
        self.mode = kwargs.get('mode')
        assert self.mode in ['TRAIN', 'TEST']
        # parse align_corners, normalization layer and activation layer cfg
        align_corners = cfg.get('align_corners', True)
        normlayer_opts = cfg.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = cfg.get('activation_opts', {'type': 'relu', 'opts': {'inplace': True}})
        if cfg['distributed']['is_on'] and 'syncbatchnorm' in normlayer_opts['type']: normlayer_opts.update({'type': 'distsyncbatchnorm'})
        self.align_corners = align_corners
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        backbone_cfg.update({'normlayer_opts': copy.deepcopy(normlayer_opts)})
        self.backbone_net = BuildBackbone(backbone_cfg)
        # build aspp net
        aspp_cfg = {
            'in_channels': cfg['aspp']['in_channels'],
            'out_channels': cfg['aspp']['out_channels'],
            'rates': cfg['aspp']['rates'],
            'align_corners': align_corners,
            'normlayer_opts': copy.deepcopy(normlayer_opts),
            'activation_opts': copy.deepcopy(activation_opts),
        }
        self.aspp_net = DepthwiseSeparableASPP(**aspp_cfg)
        # build shortcut
        shortcut_cfg = cfg['shortcut']
        self.shortcut = nn.Sequential(nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (shortcut_cfg['out_channels'], normlayer_opts['opts'])),
                                      BuildActivation(activation_opts['type'], **activation_opts['opts']))
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(DepthwiseSeparableConv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False, activation_opts=activation_opts, normlayer_opts=normlayer_opts),
                                     DepthwiseSeparableConv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False, activation_opts=activation_opts, normlayer_opts=normlayer_opts),
                                     nn.Dropout2d(decoder_cfg['dropout']),
                                     nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                                               BuildNormalizationLayer(normlayer_opts['type'], (auxiliary_cfg['out_channels'], normlayer_opts['opts'])),
                                               BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                               nn.Dropout2d(auxiliary_cfg['dropout']),
                                               nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_normlayer', False): self.freezenormlayer()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.backbone_net(x)
        # feed to aspp
        aspp_out = self.aspp_net(x4)
        aspp_out = F.interpolate(aspp_out, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=self.align_corners)
        # feed to shortcut
        shortcut_out = self.shortcut(x1)
        # feed to decoder
        features = torch.cat([aspp_out, shortcut_out], dim=1)
        preds = self.decoder(features)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(x3)
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            loss_dict = self.calculateloss({'preds': preds, 'preds_aux': preds_aux}, targets, losses_cfg)
            return loss_dict
        return preds
    '''return all layers'''
    def alllayers(self):
        return {
                'backbone_net': self.backbone_net, 
                'aspp_net': self.aspp_net, 
                'shortcut': self.shortcut,
                'decoder': self.decoder,
                'auxiliary_decoder': self.auxiliary_decoder
            }
    '''freeze normalization layer'''
    def freezenormlayer(self):
        for module in self.modules():
            if type(module) in BuildNormalizationLayer(only_get_all_supported=True):
                module.eval()
    '''calculate losses'''
    @staticmethod
    def calculateloss(preds, targets, losses_cfg):
        targets_seg = targets['segmentation']
        # calculate the auxiliary loss
        supported_losses = {
            'celoss': CrossEntropyLoss,
            'sigmoidfocalloss': SigmoidFocalLoss,
        }
        aux_cfg = losses_cfg['auxiliary']
        loss_aux = 0
        preds_aux = preds['preds_aux'].permute((0, 2, 3, 1)).contiguous()
        for key, value in aux_cfg.items():
            assert key in supported_losses, 'unsupport auxiliary loss type %s...' % key
            loss_aux += supported_losses[key](preds=preds_aux.view(-1, preds_aux.size(3)),
                                              targets=targets_seg.view(-1),
                                              scale_factor=value['scale_factor'],
                                              **value['opts'])
        # calculate the classification loss
        supported_losses = {
            'celoss': CrossEntropyLoss,
            'sigmoidfocalloss': SigmoidFocalLoss,
        }
        cls_cfg = losses_cfg['classification']
        loss_cls = 0
        preds = preds['preds'].permute((0, 2, 3, 1)).contiguous()
        for key, value in cls_cfg.items():
            assert key in supported_losses, 'unsupport classification loss type %s...' % key
            loss_cls += supported_losses[key](preds=preds.view(-1, preds.size(3)), 
                                              targets=targets_seg.view(-1), 
                                              scale_factor=value['scale_factor'],
                                              **value['opts'])
        # return the losses
        loss_dict = {'loss_aux': loss_aux, 'loss_cls': loss_cls}
        return loss_dict