'''
Function:
    Implementation of SetCriterion
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ...maskformer.transformers.criterion import SetCriterion as MaskFormerSetCriterion
from .misc import pointsample, getuncertainpointcoordswithrandomness, calculateuncertainty


'''SetCriterion'''
class SetCriterion(MaskFormerSetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio):
        super(SetCriterion, self).__init__(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses)
        # set attributes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
    '''sigmoidceloss'''
    def sigmoidceloss(self, inputs, targets, num_masks):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        return loss.mean(1).sum() / num_masks
    '''lossmasks'''
    def lossmasks(self, outputs, targets, indices, num_masks):
        assert 'pred_masks' in outputs
        src_idx = self.getsrcpermutationidx(indices)
        tgt_idx = self.gettgtpermutationidx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = self.nestedtensorfromtensorlist(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # no need to upsample predictions as we are using normalized coordinates: N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = getuncertainpointcoordswithrandomness(
                src_masks, lambda logits: calculateuncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = pointsample(target_masks, point_coords, align_corners=False).squeeze(1)
        # get logits
        point_logits = pointsample(src_masks, point_coords, align_corners=False).squeeze(1)
        # calculate losses
        losses = {
            'loss_mask': self.sigmoidceloss(point_logits, point_labels, num_masks),
            'loss_dice': self.diceloss(point_logits, point_labels, num_masks),
        }
        # return losses
        return losses