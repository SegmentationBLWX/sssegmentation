'''
Function:
    Implementation of ANNNet
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from .afnblock import AFNBlock
from .apnblock import APNBlock
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''ANNNet'''
class ANNNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ANNNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build AFNBlock
        self.afn_block = AFNBlock(
            low_in_channels=head_cfg['in_channels_list'][0], high_in_channels=head_cfg['in_channels_list'][1], transform_channels=head_cfg['transform_channels'], 
            out_channels=head_cfg['in_channels_list'][1], query_scales=head_cfg['query_scales'], key_pool_scales=head_cfg['key_pool_scales'],
            norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg),
        )
        # build APNBlock
        self.apn_block = APNBlock(
            in_channels=head_cfg['feats_channels'], transform_channels=head_cfg['transform_channels'], out_channels=head_cfg['feats_channels'], 
            query_scales=head_cfg['query_scales'], key_pool_scales=head_cfg['key_pool_scales'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg),
        )
        # build bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels_list'][1], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        low_feats, high_feats = backbone_outputs[-2], backbone_outputs[-1]
        # feed to AFNBlock
        feats = self.afn_block(low_feats, high_feats)
        feats = self.decoder[0](feats)
        # feed to bottleneck
        feats = self.bottleneck(feats)
        # feed to APNBlock
        feats = self.apn_block(feats)
        # feed to decoder
        seg_logits = self.decoder[1](feats)
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs