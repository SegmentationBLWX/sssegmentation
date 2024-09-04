'''
Function:
    Implementation of ISANet
Author:
    Zhenchao Jin
'''
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ..base import SelfAttentionBlock as _SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''SelfAttentionBlock'''
class SelfAttentionBlock(_SelfAttentionBlock):
    def __init__(self, in_channels, feats_channels, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels, query_in_channels=in_channels, transform_channels=feats_channels, out_channels=in_channels, share_key_query=False,
            query_downsample=None, key_downsample=None, key_query_num_convs=2, key_query_norm=True, value_out_num_convs=1, value_out_norm=False,
            matmul_norm=True, with_out_project=False, norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg)
        )
        self.output_project = self.buildproject(
            in_channels=in_channels, out_channels=in_channels, num_convs=1, use_norm=True, norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        context = super(SelfAttentionBlock, self).forward(x, x)
        return self.output_project(context)


'''ISANet'''
class ISANet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ISANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build isa module
        self.down_factor = head_cfg['down_factor']
        self.in_conv = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.global_relation = SelfAttentionBlock(
            in_channels=head_cfg['feats_channels'], feats_channels=head_cfg['isa_channels'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg)
        )
        self.local_relation = SelfAttentionBlock(
            in_channels=head_cfg['feats_channels'], feats_channels=head_cfg['isa_channels'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] * 2, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
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
        # feed to isa module
        feats = self.in_conv(backbone_outputs[-1])
        residual = feats
        n, c, h, w = feats.size()
        loc_h, loc_w = self.down_factor
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            feats = F.pad(feats, padding)
        # --global relation
        feats = feats.view(n, c, glb_h, loc_h, glb_w, loc_w)
        # ----do permutation to gather global group
        feats = feats.permute(0, 3, 5, 1, 2, 4)
        feats = feats.reshape(-1, c, glb_h, glb_w)
        # ----apply attention within each global group
        feats = self.global_relation(feats)
        # --local relation
        feats = feats.view(n, loc_h, loc_w, c, glb_h, glb_w)
        # ----do permutation to gather local group
        feats = feats.permute(0, 4, 5, 3, 1, 2)
        feats = feats.reshape(-1, c, loc_h, loc_w)
        # ----apply attention within each local group
        feats = self.local_relation(feats)
        # --permute each pixel back to its original position
        feats = feats.view(n, glb_h, glb_w, c, loc_h, loc_w)
        feats = feats.permute(0, 3, 1, 4, 2, 5)
        feats = feats.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2: pad_h//2+h, pad_w//2: pad_w//2+w]
        feats = self.out_conv(torch.cat([feats, residual], dim=1))
        # feed to decoder
        seg_logits = self.decoder(feats)
        # return according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs