'''
Function:
    Implementation of PointRend
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base import FPN, BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization
try:
    from mmcv.ops import point_sample as PointSample
except:
    PointSample = None


'''PointRend'''
class PointRend(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(PointRend, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build fpn
        self.fpn_neck = FPN(
            in_channels_list=head_cfg['fpn_in_channels_list'], out_channels=head_cfg['feats_channels'], upsample_cfg=head_cfg['upsample_cfg'],
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.scale_heads, feature_stride_list = nn.ModuleList(), head_cfg['feature_stride_list']
        for i in range(len(feature_stride_list)):
            head_length = max(1, int(np.log2(feature_stride_list[i]) - np.log2(feature_stride_list[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(nn.Sequential(
                    nn.Conv2d(head_cfg['feats_channels'] if k == 0 else head_cfg['scale_head_channels'], head_cfg['scale_head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(placeholder=head_cfg['scale_head_channels'], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
                if feature_stride_list[i] != feature_stride_list[0]:
                    scale_head.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        # point rend
        self.num_fcs, self.coarse_pred_each_layer = head_cfg['num_fcs'], head_cfg['coarse_pred_each_layer']
        fc_in_channels = sum(head_cfg['pointrend_in_channels_list']) + cfg['num_classes']
        fc_channels = head_cfg['feats_channels']
        self.fcs = nn.ModuleList()
        for k in range(self.num_fcs):
            fc = nn.Sequential(
                nn.Conv1d(fc_in_channels, fc_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=fc_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += cfg['num_classes'] if self.coarse_pred_each_layer else 0
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout(head_cfg['dropout']),
            nn.Conv1d(fc_in_channels, cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        assert (cfg['auxiliary'] is not None) and isinstance(cfg['auxiliary'], dict), 'auxiliary must be given and only support dict type'
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to fpn
        fpn_outs = self.fpn_neck(list(backbone_outputs))
        feats = self.scale_heads[0](fpn_outs[0])
        for i in range(1, len(self.cfg['head']['feature_stride_list'])):
            feats = feats + F.interpolate(self.scale_heads[i](fpn_outs[i]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to auxiliary decoder
        seg_logits_aux = self.auxiliary_decoder(feats)
        feats = fpn_outs[0]
        # if mode is TRAIN or TRAIN_DEVELOP
        ssseg_outputs = SSSegOutputStructure(mode=self.mode, auto_validate=False)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            with torch.no_grad():
                points = self.getpointstrain(seg_logits_aux, self.calculateuncertainty, cfg=self.cfg['head']['train'])
            fine_grained_point_feats = self.getfinegrainedpointfeats([feats], points)
            coarse_point_feats = self.getcoarsepointfeats(seg_logits_aux, points)
            feats_concat = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
            for fc in self.fcs:
                feats_concat = fc(feats_concat)
                if self.coarse_pred_each_layer:
                    feats_concat = torch.cat([feats_concat, coarse_point_feats], dim=1)
            seg_logits = self.decoder(feats_concat)
            point_labels = PointSample(data_meta.gettargets()['seg_targets'].unsqueeze(1).float(), points, mode='nearest', align_corners=self.align_corners)
            point_labels = point_labels.squeeze(1).long()
            targets = data_meta.gettargets()
            targets['point_labels'] = point_labels
            seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
            loss, losses_log_dict = self.calculatelosses(
                predictions={'loss_cls': seg_logits, 'loss_aux': seg_logits_aux}, targets=targets, losses_cfg=self.cfg['losses'], map_preds_to_tgts_dict={'loss_cls': 'point_labels', 'loss_aux': 'seg_targets'}
            )
            ssseg_outputs.setvariable('loss', loss)
            ssseg_outputs.setvariable('losses_log_dict', losses_log_dict)
            if self.mode in ['TRAIN']: return ssseg_outputs
        # if mode is TEST
        refined_seg_logits = seg_logits_aux.clone()
        for _ in range(self.cfg['head']['test']['subdivision_steps']):
            refined_seg_logits = F.interpolate(
                input=refined_seg_logits, scale_factor=self.cfg['head']['test']['scale_factor'], mode='bilinear', align_corners=self.align_corners
            )
            batch_size, channels, height, width = refined_seg_logits.shape
            point_indices, points = self.getpointstest(refined_seg_logits, self.calculateuncertainty, cfg=self.cfg['head']['test'])
            fine_grained_point_feats = self.getfinegrainedpointfeats([feats], points)
            coarse_point_feats = self.getcoarsepointfeats(seg_logits_aux, points)
            feats_concat = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
            for fc in self.fcs:
                feats_concat = fc(feats_concat)
                if self.coarse_pred_each_layer:
                    feats_concat = torch.cat([feats_concat, coarse_point_feats], dim=1)
            seg_logits = self.decoder(feats_concat)
            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_seg_logits = refined_seg_logits.reshape(batch_size, channels, height * width)
            refined_seg_logits = refined_seg_logits.scatter_(2, point_indices, seg_logits)
            refined_seg_logits = refined_seg_logits.view(batch_size, channels, height, width)
        ssseg_outputs.setvariable('seg_logits', refined_seg_logits)
        return ssseg_outputs
    '''sample from coarse grained features'''
    def getcoarsepointfeats(self, seg_logits, points):
        coarse_feats = PointSample(seg_logits, points, align_corners=self.align_corners)
        return coarse_feats
    '''sample from fine grained features'''
    def getfinegrainedpointfeats(self, x, points):
        fine_grained_feats_list = [PointSample(_, points, align_corners=self.align_corners) for _ in x]
        if len(fine_grained_feats_list) > 1:
            fine_grained_feats = torch.cat(fine_grained_feats_list, dim=1)
        else:
            fine_grained_feats = fine_grained_feats_list[0]
        return fine_grained_feats
    '''estimate uncertainty based on seg logits'''
    @staticmethod
    def calculateuncertainty(seg_logits):
        top2_scores = torch.topk(seg_logits, k=2, dim=1)[0]
        return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)
    '''sample points for training'''
    def getpointstrain(self, seg_logits, uncertainty_func, cfg):
        # set attrs
        num_points = cfg['num_points']
        oversample_ratio = cfg['oversample_ratio']
        importance_sample_ratio = cfg['importance_sample_ratio']
        assert (oversample_ratio >= 1) and (0 <= importance_sample_ratio <= 1)
        # sample
        batch_size = seg_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(batch_size, num_sampled, 2, device=seg_logits.device)
        point_logits = PointSample(seg_logits, point_coords)
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(batch_size, dtype=torch.long, device=seg_logits.device)
        idx = idx + shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(batch_size, num_uncertain_points, 2)
        if num_random_points > 0:
            rand_point_coords = torch.rand(batch_size, num_random_points, 2, device=seg_logits.device)
            point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
        # return
        return point_coords
    '''sample points for testing'''
    def getpointstest(self, seg_logits, uncertainty_func, cfg):
        num_points = cfg['subdivision_num_points']
        uncertainty_map = uncertainty_func(seg_logits)
        batch_size, _, height, width = uncertainty_map.shape
        h_step, w_step = 1.0 / height, 1.0 / width
        uncertainty_map = uncertainty_map.view(batch_size, height * width)
        num_points = min(height * width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        point_coords = torch.zeros(batch_size, num_points, 2, dtype=torch.float, device=seg_logits.device)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices % width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices // width).float() * h_step
        return point_indices, point_coords