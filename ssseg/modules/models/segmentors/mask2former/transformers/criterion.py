'''
Function:
    Implementation of SetCriterion
Author:
    Zhenchao Jin
'''
import torch.nn.functional as F
from ....samplers import BuildPixelSampler
from ...maskformer.transformers.criterion import SetCriterion as MaskFormerSetCriterion


'''SetCriterion'''
class SetCriterion(MaskFormerSetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, pixelsampler_cfg):
        super(SetCriterion, self).__init__(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses)
        # set attributes
        self.pixel_sampler = BuildPixelSampler(pixelsampler_cfg=pixelsampler_cfg)
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
        # pixel sampling
        point_logits, point_labels = self.pixel_sampler.sample(src_masks, target_masks)
        # calculate losses
        losses = {
            'loss_mask': self.sigmoidceloss(point_logits, point_labels, num_masks),
            'loss_dice': self.diceloss(point_logits, point_labels, num_masks),
        }
        # return losses
        return losses