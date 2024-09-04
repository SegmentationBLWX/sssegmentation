'''
Function:
    Implementation of OCRNet
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule
from ...backbones import BuildActivation, BuildNormalization


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
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': head_cfg['scale']
        }
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        # build object context block
        self.object_context_block = ObjectContextBlock(
            in_channels=head_cfg['feats_channels'], transform_channels=head_cfg['transform_channels'], scale=head_cfg['scale'],
            align_corners=align_corners, norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to auxiliary decoder
        seg_logits_aux = self.auxiliary_decoder(backbone_outputs[-2])
        # feed to bottleneck
        feats = self.bottleneck(backbone_outputs[-1])
        # feed to ocr module
        context = self.spatial_gather_module(feats, seg_logits_aux)
        feats = self.object_context_block(feats, context)
        # feed to decoder
        seg_logits = self.decoder(feats)
        # return according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
            seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
            loss, losses_log_dict = self.calculatelosses(
                predictions={'loss_cls': seg_logits, 'loss_aux': seg_logits_aux}, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses']
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs