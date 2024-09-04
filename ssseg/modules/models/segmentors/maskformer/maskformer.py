'''
Function:
    Implementation of MaskFormer
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..base import BaseSegmentor
from ..pspnet import PyramidPoolingModule
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization
from .transformers import Predictor, SetCriterion, HungarianMatcher


'''MaskFormer'''
class MaskFormer(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(MaskFormer, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'pool_scales': head_cfg['pool_scales'],
            'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build lateral convs
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
        self.lateral_convs = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list'][:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg_copy),
            ))
        # build fpn convs
        self.fpn_convs = nn.ModuleList()
        for in_channels in [head_cfg['feats_channels'], ] * len(self.lateral_convs):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg_copy),
            ))
        # build decoder
        self.decoder_mask = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['mask_feats_channels'], kernel_size=3, stride=1, padding=1)
        )
        head_cfg['predictor']['num_classes'] = cfg['num_classes']
        head_cfg['predictor']['mask_dim'] = head_cfg['mask_feats_channels']
        head_cfg['predictor']['in_channels'] = head_cfg['in_channels_list'][-1]
        self.decoder_predictor = Predictor(**head_cfg['predictor'])
        # build matcher and criterion
        matcher = HungarianMatcher(**head_cfg['matcher'])
        weight_dict = {'loss_ce': head_cfg['matcher']['cost_class'], 'loss_mask': head_cfg['matcher']['cost_mask'], 'loss_dice': head_cfg['matcher']['cost_dice']}
        if head_cfg['predictor']['deep_supervision']:
            dec_layers = head_cfg['predictor']['dec_layers']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(cfg['num_classes'], matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'masks'])
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(backbone_outputs[-1])
        # apply fpn
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        p1, p2, p3, p4 = lateral_outputs
        fpn_out = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=self.align_corners) + p3
        fpn_out = self.fpn_convs[0](fpn_out)
        fpn_out = F.interpolate(fpn_out, size=p2.shape[2:], mode='bilinear', align_corners=self.align_corners) + p2
        fpn_out = self.fpn_convs[1](fpn_out)
        fpn_out = F.interpolate(fpn_out, size=p1.shape[2:], mode='bilinear', align_corners=self.align_corners) + p1
        fpn_out = self.fpn_convs[2](fpn_out)
        # feed to decoder
        mask_features = self.decoder_mask(fpn_out)
        predictions = self.decoder_predictor(backbone_outputs[-1], mask_features)
        # forward according to the mode
        ssseg_outputs = SSSegOutputStructure(mode=self.mode, auto_validate=False)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            losses_dict = self.criterion(predictions, data_meta.gettargets())
            for k in list(losses_dict.keys()):
                if k in self.criterion.weight_dict:
                    losses_dict[k] *= self.criterion.weight_dict[k]
                else:
                    losses_dict.pop(k)
            loss, losses_log_dict = 0, {}
            for loss_key, loss_value in losses_dict.items():
                loss_value = loss_value.mean()
                loss = loss + loss_value
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
                losses_log_dict[loss_key] = loss_value.item()
            losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
            ssseg_outputs.setvariable('loss', loss)
            ssseg_outputs.setvariable('losses_log_dict', losses_log_dict)
            if self.mode in ['TRAIN']: return ssseg_outputs
        mask_cls_results = predictions['pred_logits']
        mask_pred_results = predictions['pred_masks']
        mask_pred_results = F.interpolate(mask_pred_results, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = []
        for mask_cls, mask_pred in zip(mask_cls_results, mask_pred_results):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
            predictions.append(semseg.unsqueeze(0))
        seg_logits = torch.cat(predictions, dim=0)
        ssseg_outputs.setvariable('seg_logits', seg_logits)            
        return ssseg_outputs