'''
Function:
    Implementation of OCRNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''OCRNet'''
class OCRNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(OCRNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build auxiliary decoder
        assert (cfg['auxiliary'] is not None) and isinstance(cfg['auxiliary'], dict), 'auxiliary must be given and only support dict type'
        self.setauxiliarydecoder(cfg['auxiliary'])
        # build bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': head_cfg['scale']
        }
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        # build object context block
        self.object_context_block = ObjectContextBlock(
            in_channels=head_cfg['feats_channels'], 
            transform_channels=head_cfg['transform_channels'], 
            scale=head_cfg['scale'],
            align_corners=align_corners,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'bottleneck', 'spatial_gather_module', 'object_context_block', 'decoder', 'auxiliary_decoder']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to auxiliary decoder
        predictions_aux = self.auxiliary_decoder(backbone_outputs[-2])
        # feed to bottleneck
        feats = self.bottleneck(backbone_outputs[-1])
        # feed to ocr module
        context = self.spatial_gather_module(feats, predictions_aux)
        feats = self.object_context_block(feats, context)
        # feed to decoder
        predictions = self.decoder(feats)
        # return according to the mode
        if self.mode == 'TRAIN':
            predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': predictions, 'loss_aux': predictions_aux}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return predictions