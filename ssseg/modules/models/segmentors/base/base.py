'''
Function:
    Base segmentor for all supported segmentors
Author:
    Zhenchao Jin
'''
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ...losses import calculatelosses
from ....utils import SSSegInputStructure
from ...samplers import BuildPixelSampler
from .auxiliary import BuildAuxiliaryDecoder
from ...backbones import BuildBackbone, NormalizationBuilder


'''BaseSegmentor'''
class BaseSegmentor(nn.Module):
    def __init__(self, cfg, mode):
        super(BaseSegmentor, self).__init__()
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['TRAIN', 'TEST', 'TRAIN_DEVELOP']
        # parse align_corners, normalization layer and activation layer cfg
        for key in ['align_corners', 'norm_cfg', 'act_cfg']:
            if key in cfg: setattr(self, key, cfg[key])
        # build backbone
        self.setbackbone(cfg=cfg)
        # build pixel sampler
        self.setpixelsampler(cfg=cfg)
    '''forward'''
    def forward(self, data_meta):
        raise NotImplementedError('not to be implemented')
    '''customizepredsandlosses'''
    def customizepredsandlosses(self, seg_logits, annotations, backbone_outputs, losses_cfg, img_size, auto_calc_loss=True, preds_to_tgts_mapping=None):
        seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = {'loss_cls': seg_logits}
        if hasattr(self, 'auxiliary_decoder'):
            backbone_outputs = backbone_outputs[:-1]
            if isinstance(self.auxiliary_decoder, nn.ModuleList):
                assert len(backbone_outputs) >= len(self.auxiliary_decoder)
                backbone_outputs = backbone_outputs[-len(self.auxiliary_decoder):]
                for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoder)):
                    predictions[f'loss_aux{idx+1}'] = F.interpolate(dec(out), size=img_size, mode='bilinear', align_corners=self.align_corners)
            else:
                predictions['loss_aux'] = F.interpolate(self.auxiliary_decoder(backbone_outputs[-1]), size=img_size, mode='bilinear', align_corners=self.align_corners)
        if not auto_calc_loss: return predictions
        return calculatelosses(predictions=predictions, annotations=annotations, losses_cfg=losses_cfg, preds_to_tgts_mapping=preds_to_tgts_mapping, pixel_sampler=self.pixel_sampler)
    '''inference'''
    def inference(self, images, forward_args=None):
        # assert and initialize
        inference_cfg = self.cfg['inference']
        assert inference_cfg['forward']['mode'] in ['whole', 'slide']
        use_probs_before_resize = inference_cfg['tta']['use_probs_before_resize']
        images = images.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        if forward_args is None: forward_args = {}
        # inference
        if inference_cfg['forward']['mode'] == 'whole':
            seg_logits = self(SSSegInputStructure(images=images, mode=self.mode), **forward_args).seg_logits
            if use_probs_before_resize:
                seg_logits = F.softmax(seg_logits, dim=1)
        else:
            stride_h, stride_w = inference_cfg['forward']['stride']
            cropsize_h, cropsize_w = inference_cfg['forward']['cropsize']
            batch_size, _, image_h, image_w = images.size()
            num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
            num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
            seg_logits = images.new_zeros((batch_size, self.cfg['num_classes'], image_h, image_w))
            count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
            for h_idx in range(num_grids_h):
                for w_idx in range(num_grids_w):
                    x1, y1 = w_idx * stride_w, h_idx * stride_h
                    x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                    x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                    crop_images = images[:, :, y1:y2, x1:x2]
                    seg_logits_crop = self(SSSegInputStructure(images=crop_images, mode=self.mode), **forward_args).seg_logits
                    seg_logits_crop = F.interpolate(seg_logits_crop, size=crop_images.size()[2:], mode='bilinear', align_corners=self.align_corners)
                    if use_probs_before_resize:
                        seg_logits_crop = F.softmax(seg_logits_crop, dim=1)
                    seg_logits += F.pad(seg_logits_crop, (int(x1), int(seg_logits.shape[3] - x2), int(y1), int(seg_logits.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            seg_logits = seg_logits / count_mat
        # return seg_logits
        return seg_logits
    '''auginference'''
    def auginference(self, images, forward_args=None):
        # initialize
        inference_cfg = self.cfg['inference']
        infer_tta_cfg, seg_logits_list = inference_cfg['tta'], []
        # iter to inference
        for scale_factor in infer_tta_cfg['multiscale']:
            images_scale = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=self.align_corners)
            seg_logits = self.inference(images=images_scale, forward_args=forward_args).cpu()
            seg_logits_list.append(seg_logits)
            if infer_tta_cfg['flip']:
                images_scale_flip = torch.from_numpy(np.flip(images_scale.cpu().numpy(), axis=3).copy())
                seg_logits_flip = self.inference(images=images_scale_flip, forward_args=forward_args)
                fixed_seg_target_pairs = infer_tta_cfg.get('fixed_seg_target_pairs', None)
                if fixed_seg_target_pairs is None:
                    for data_pipeline in self.cfg['dataset']['train']['data_pipelines']:
                        if 'RandomFlip' in data_pipeline:
                            if isinstance(data_pipeline, dict):
                                fixed_seg_target_pairs = data_pipeline['RandomFlip'].get('fixed_seg_target_pairs', None)
                            else:
                                fixed_seg_target_pairs = data_pipeline[-1].get('fixed_seg_target_pairs', None)
                if fixed_seg_target_pairs is not None:
                    seg_logits_flip_clone = seg_logits_flip.data.clone()
                    for (pair_a, pair_b) in fixed_seg_target_pairs:
                        seg_logits_flip[:, pair_a, :, :] = seg_logits_flip_clone[:, pair_b, :, :]
                        seg_logits_flip[:, pair_b, :, :] = seg_logits_flip_clone[:, pair_a, :, :]
                seg_logits_flip = torch.from_numpy(np.flip(seg_logits_flip.cpu().numpy(), axis=3).copy()).type_as(seg_logits)
                seg_logits_list.append(seg_logits_flip)
        # return seg_logits_list
        return seg_logits_list
    '''transforminputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['type'] in ['HRNet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        return outs
    '''setauxiliarydecoder'''
    def setauxiliarydecoder(self, auxiliary_cfg):
        if auxiliary_cfg is None: return
        if isinstance(auxiliary_cfg, dict): auxiliary_cfg = [auxiliary_cfg]
        self.auxiliary_decoder = nn.ModuleList()
        for aux_cfg in auxiliary_cfg:
            self.auxiliary_decoder.append(BuildAuxiliaryDecoder(auxiliary_cfg=aux_cfg, norm_cfg=self.norm_cfg.copy(), act_cfg=self.act_cfg.copy(), num_classes=self.cfg['num_classes']))
        if len(self.auxiliary_decoder) == 1: self.auxiliary_decoder = self.auxiliary_decoder[0]
    '''setbackbone'''
    def setbackbone(self, cfg):
        if 'backbone' not in cfg: return
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    '''setpixelsampler'''
    def setpixelsampler(self, cfg):
        if 'pixelsampler' in cfg['head']:
            self.pixel_sampler = BuildPixelSampler(cfg['head']['pixelsampler'])
        else:
            self.pixel_sampler = None
    '''freezenormalization'''
    def freezenormalization(self, norm_list=None):
        if norm_list is None:
            norm_list = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        for module in self.modules():
            if NormalizationBuilder.isnorm(module, norm_list):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
    '''train'''
    def train(self, mode=True):
        self.mode = 'TRAIN' if mode else 'TEST'
        return super().train(mode)
    '''eval'''
    def eval(self):
        self.mode = 'TEST'
        return super().eval()