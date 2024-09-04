'''
Function:
    Implementation of Mask2Former
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from .transformers import MultiScaleMaskedTransformerDecoder, MSDeformAttnPixelDecoder, SetCriterion, HungarianMatcher


'''ShapeSpec'''
class ShapeSpec():
    def __init__(self, stride, channels):
        self.stride = stride
        self.channels = channels


'''Mask2Former'''
class Mask2Former(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(Mask2Former, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build pixel decoder
        iterator = zip(head_cfg['pixel_decoder']['input_shape']['strides'], head_cfg['pixel_decoder']['input_shape']['in_channels'])
        assert len(head_cfg['pixel_decoder']['input_shape']['strides']) == 4
        head_cfg['pixel_decoder']['input_shape'] = {f'res{idx+2}': ShapeSpec(stride, channels) for idx, (stride, channels) in enumerate(iterator)}
        self.pixel_decoder = MSDeformAttnPixelDecoder(**head_cfg['pixel_decoder'])
        # build predictor
        predictor_cfg = copy.deepcopy(head_cfg['predictor'])
        predictor_cfg['dec_layers'] = predictor_cfg['dec_layers'] - 1
        self.predictor = MultiScaleMaskedTransformerDecoder(num_classes=cfg['num_classes'], **predictor_cfg)
        # build matcher and criterion
        matcher = HungarianMatcher(**head_cfg['matcher'])
        weight_dict = {'loss_ce': head_cfg['matcher']['cost_class'], 'loss_mask': head_cfg['matcher']['cost_mask'], 'loss_dice': head_cfg['matcher']['cost_dice']}
        if head_cfg['deep_supervision']:
            dec_layers, aux_weight_dict = head_cfg['predictor']['dec_layers'], {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(cfg['num_classes'], matcher=matcher, weight_dict=weight_dict, **head_cfg['criterion'])
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        from torch.cuda.amp import autocast
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to pixel decoder
        assert len(backbone_outputs) == 4
        features = {
            'res2': backbone_outputs[0], 'res3': backbone_outputs[1], 'res4': backbone_outputs[2], 'res5': backbone_outputs[3]
        }
        with autocast(enabled=False):
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forwardfeatures(features)
        # feed to predictor
        predictions = self.predictor(multi_scale_features, mask_features, None)
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