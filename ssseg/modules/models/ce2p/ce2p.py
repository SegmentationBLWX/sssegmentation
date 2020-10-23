'''
Function:
    Implementation of CE2P
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...losses import *
from ...backbones import *
from .epm import EdgePerceivingModule
from ..pspnet import PyramidPoolingModule


'''ce2p'''
class CE2P(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CE2P, self).__init__()
        self.mode = kwargs.get('mode')
        assert self.mode in ['TRAIN', 'TEST']
        # parse align_corners, normalization layer and activation layer cfg
        align_corners = cfg.get('align_corners', True)
        normlayer_opts = cfg.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}})
        activation_opts = cfg.get('activation_opts', {'type': 'leakyrelu', 'opts': {'negative_slope': 0.01, 'inplace': True}})
        if cfg['distributed']['is_on'] and 'syncbatchnorm' in normlayer_opts['type']: normlayer_opts.update({'type': 'distsyncbatchnorm'})
        self.align_corners = align_corners
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        backbone_cfg.update({'normlayer_opts': copy.deepcopy(normlayer_opts)})
        self.backbone_net = BuildBackbone(backbone_cfg)
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': cfg['ppm']['in_channels'],
            'out_channels': cfg['ppm']['out_channels'],
            'bin_sizes': cfg['ppm']['bin_sizes'],
            'align_corners': align_corners,
            'normlayer_opts': copy.deepcopy(normlayer_opts),
            'activation_opts': copy.deepcopy(activation_opts),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        lateral_ppm_cfg = cfg['lateral_ppm']
        self.lateral_ppm_layer = nn.Sequential(nn.Conv2d(lateral_ppm_cfg['in_channels'], lateral_ppm_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                                               BuildNormalizationLayer(normlayer_opts['type'], (lateral_ppm_cfg['out_channels'], normlayer_opts['opts'])),
                                               BuildActivation(activation_opts['type'], **activation_opts['opts']))
        # build edge perceiving module
        epm_cfg = {
            'in_channels_list': cfg['epm']['in_channels_list'],
            'hidden_channels': cfg['epm']['hidden_channels'],
            'out_channels': cfg['epm']['out_channels'],
            'align_corners': align_corners,
            'normlayer_opts': copy.deepcopy(normlayer_opts),
            'activation_opts': copy.deepcopy(activation_opts),
        }
        self.edge_net = EdgePerceivingModule(**epm_cfg)
        # build shortcut
        shortcut_cfg = cfg['shortcut']
        self.shortcut = nn.Sequential(nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (shortcut_cfg['out_channels'], normlayer_opts['opts'])),
                                      BuildActivation(activation_opts['type'], **activation_opts['opts']))
        # build decoder stage1
        decoder_cfg = cfg['decoder']['stage1']
        self.decoder_stage1 = nn.Sequential(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                                            BuildNormalizationLayer(normlayer_opts['type'], (decoder_cfg['out_channels'], normlayer_opts['opts'])),
                                            BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                            nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                                            BuildNormalizationLayer(normlayer_opts['type'], (decoder_cfg['out_channels'], normlayer_opts['opts'])),
                                            BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                            nn.Dropout2d(decoder_cfg['dropout']), 
                                            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        # build decoder stage1
        decoder_cfg = cfg['decoder']['stage2']
        self.decoder_stage2 = nn.Sequential(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                                            BuildNormalizationLayer(normlayer_opts['type'], (decoder_cfg['out_channels'], normlayer_opts['opts'])),
                                            BuildActivation(activation_opts['type'], **activation_opts['opts']),
                                            nn.Dropout2d(decoder_cfg['dropout']), 
                                            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_normlayer', False): self.freezenormlayer()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.backbone_net(x)
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(x4)
        ppm_out = self.lateral_ppm_layer(ppm_out)
        ppm_out = F.interpolate(ppm_out, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=self.align_corners)
        # feed to edge perceiving module
        edge, edge_feats = self.edge_net((x1, x2, x3))
        edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        # feed to shortcut
        shortcut_out = self.shortcut(x1)
        # feed to decoder stage1
        features = torch.cat([ppm_out, shortcut_out], dim=1)
        features = self.decoder_stage1[:-1](features)
        preds_stage1 = self.decoder_stage1[-1](features)
        preds_stage1 = F.interpolate(preds_stage1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        # feed to decoder stage2
        features = torch.cat([features, edge_feats], dim=1)
        preds_stage2 = self.decoder_stage2(features)
        # return according to the mode
        if self.mode == 'TRAIN':
            preds_stage2 = F.interpolate(preds_stage2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            loss_dict = self.calculateloss({'preds_stage1': preds_stage1, 'preds_stage2': preds_stage2, 'edge': edge}, targets, losses_cfg)
            return loss_dict
        return preds_stage2
    '''return all layers'''
    def alllayers(self):
        return {
                'backbone_net': self.backbone_net,
                'ppm_net': self.ppm_net,
                'lateral_ppm_layer': self.lateral_ppm_layer,
                'edge_net': self.edge_net,
                'shortcut': self.shortcut,
                'decoder_stage1': self.decoder_stage1,
                'decoder_stage2': self.decoder_stage2,
            }
    '''freeze normalization layer'''
    def freezenormlayer(self):
        for module in self.modules():
            if type(module) in BuildNormalizationLayer(only_get_all_supported=True):
                module.eval()
    '''calculate losses'''
    @staticmethod
    def calculateloss(preds, targets, losses_cfg):
        supported_losses = {
            'celoss': CrossEntropyLoss,
            'sigmoidfocalloss': SigmoidFocalLoss,
        }
        # calculate the classification loss
        targets_seg = targets['segmentation']
        # --stage1
        cls_cfg = losses_cfg['classification']['stage1']
        loss_cls_stage1 = 0
        preds_stage1 = preds['preds_stage1'].permute((0, 2, 3, 1)).contiguous()
        for key, value in cls_cfg.items():
            assert key in supported_losses, 'unsupport classification loss type %s...' % key
            loss_cls_stage1 += supported_losses[key](preds=preds_stage1.view(-1, preds_stage1.size(3)), 
                                                     targets=targets_seg.view(-1), 
                                                     scale_factor=value['scale_factor'],
                                                     **value['opts'])
        # --stage2
        cls_cfg = losses_cfg['classification']['stage2']
        loss_cls_stage2 = 0
        preds_stage2 = preds['preds_stage2'].permute((0, 2, 3, 1)).contiguous()
        for key, value in cls_cfg.items():
            assert key in supported_losses, 'unsupport classification loss type %s...' % key
            loss_cls_stage2 += supported_losses[key](preds=preds_stage2.view(-1, preds_stage2.size(3)), 
                                                     targets=targets_seg.view(-1), 
                                                     scale_factor=value['scale_factor'],
                                                     **value['opts'])
        # calculate the edge loss
        targets_edge = targets['edge']
        num_neg, num_pos = torch.sum(targets_edge == 0, dtype=torch.float), torch.sum(targets_edge == 1, dtype=torch.float)
        weight_pos, weight_neg = num_neg / (num_pos + num_neg), num_pos / (num_pos + num_neg)
        weight = torch.Tensor([weight_neg, weight_pos]).type_as(targets_edge)
        edge_cfg = losses_cfg['edge']
        loss_edge = 0
        preds_edge = preds['edge'].permute((0, 2, 3, 1)).contiguous()
        for key, value in edge_cfg.items():
            assert key in supported_losses, 'unsupport edge loss type %s...' % key
            value['opts'].update({'weight': weight})
            loss_edge += supported_losses[key](preds=preds_edge.view(-1, preds_edge.size(3)), 
                                               targets=targets_edge.view(-1), 
                                               scale_factor=value['scale_factor'],
                                               **value['opts'])
        # return the losses
        loss_dict = {'loss_cls_stage1': loss_cls_stage1, 'loss_cls_stage2': loss_cls_stage2, 'loss_edge': loss_edge}
        return loss_dict