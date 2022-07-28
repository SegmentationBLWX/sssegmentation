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
from ..base import FPN, BaseModel
from mmcv.ops import point_sample as PointSample
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''PointRend'''
class PointRend(BaseModel):
    def __init__(self, cfg, mode):
        super(PointRend, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build fpn
        fpn_cfg = cfg['fpn']
        self.fpn_neck = FPN(
            in_channels_list=fpn_cfg['in_channels_list'],
            out_channels=fpn_cfg['out_channels'],
            upsample_cfg=fpn_cfg['upsample_cfg'],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.scale_heads, feature_stride_list = nn.ModuleList(), fpn_cfg['feature_stride_list']
        for i in range(len(feature_stride_list)):
            head_length = max(1, int(np.log2(feature_stride_list[i]) - np.log2(feature_stride_list[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(nn.Sequential(
                    nn.Conv2d(fpn_cfg['out_channels'] if k == 0 else fpn_cfg['scale_head_channels'], fpn_cfg['scale_head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=fpn_cfg['scale_head_channels'], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg),
                ))
                if feature_stride_list[i] != feature_stride_list[0]:
                    scale_head.append(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))
        # point rend
        pointrend_cfg = cfg['pointrend']
        self.num_fcs, self.coarse_pred_each_layer = pointrend_cfg['num_fcs'], pointrend_cfg['coarse_pred_each_layer']
        fc_in_channels = sum(pointrend_cfg['in_channels_list']) + cfg['num_classes']
        fc_channels = pointrend_cfg['feats_channels']
        self.fcs = nn.ModuleList()
        for k in range(self.num_fcs):
            fc = nn.Sequential(
                nn.Conv1d(fc_in_channels, fc_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(placeholder=fc_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += cfg['num_classes'] if self.coarse_pred_each_layer else 0
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout(decoder_cfg['dropout']),
            nn.Conv1d(fc_in_channels, cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        assert (cfg['auxiliary'] is not None) and isinstance(cfg['auxiliary'], dict), 'auxiliary must be given and only support dict type'
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to fpn
        fpn_outs = self.fpn_neck(list(backbone_outputs))
        feats = self.scale_heads[0](fpn_outs[0])
        for i in range(1, len(self.cfg['fpn']['feature_stride_list'])):
            feats = feats + F.interpolate(self.scale_heads[i](fpn_outs[i]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to auxiliary decoder
        predictions_aux = self.auxiliary_decoder(feats)
        feats = fpn_outs[0]
        # if mode is TRAIN
        if self.mode == 'TRAIN':
            with torch.no_grad():
                points = self.getpointstrain(predictions_aux, self.calculateuncertainty, cfg=self.cfg['pointrend']['train'])
            fine_grained_point_feats = self.getfinegrainedpointfeats([feats], points)
            coarse_point_feats = self.getcoarsepointfeats(predictions_aux, points)
            outputs = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
            for fc in self.fcs:
                outputs = fc(outputs)
                if self.coarse_pred_each_layer:
                    outputs = torch.cat([outputs, coarse_point_feats], dim=1)
            predictions = self.decoder(outputs)
            point_labels = PointSample(targets['segmentation'].unsqueeze(1).float(), points, mode='nearest', align_corners=self.align_corners)
            point_labels = point_labels.squeeze(1).long()
            targets['point_labels'] = point_labels
            predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': predictions, 'loss_aux': predictions_aux}, 
                targets=targets,
                losses_cfg=losses_cfg,
                map_preds_to_tgts_dict={'loss_cls': 'point_labels', 'loss_aux': 'segmentation'}
            )
        # if mode is TEST
        refined_seg_logits = predictions_aux.clone()
        for _ in range(self.cfg['pointrend']['test']['subdivision_steps']):
            refined_seg_logits = F.interpolate(
                input=refined_seg_logits, 
                scale_factor=self.cfg['pointrend']['test']['scale_factor'],
                mode='bilinear',
                align_corners=self.align_corners
            )
            batch_size, channels, height, width = refined_seg_logits.shape
            point_indices, points = self.getpointstest(refined_seg_logits, self.calculateuncertainty, cfg=self.cfg['pointrend']['test'])
            fine_grained_point_feats = self.getfinegrainedpointfeats([feats], points)
            coarse_point_feats = self.getcoarsepointfeats(predictions_aux, points)
            outputs = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
            for fc in self.fcs:
                outputs = fc(outputs)
                if self.coarse_pred_each_layer:
                    outputs = torch.cat([outputs, coarse_point_feats], dim=1)
            predictions = self.decoder(outputs)
            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_seg_logits = refined_seg_logits.reshape(batch_size, channels, height * width)
            refined_seg_logits = refined_seg_logits.scatter_(2, point_indices, predictions)
            refined_seg_logits = refined_seg_logits.view(batch_size, channels, height, width)
        return refined_seg_logits
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
        idx += shift[:, None]
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
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'fpn_neck': self.fpn_neck,
            'scale_heads': self.scale_heads,
            'fcs': self.fcs,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }