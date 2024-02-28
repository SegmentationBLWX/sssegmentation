'''
Function:
    Implementation of OHEMPixelSampler
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F
from .base import BasePixelSampler


'''OHEMPixelSampler'''
class OHEMPixelSampler(BasePixelSampler):
    def __init__(self, loss_func=None, thresh=None, min_kept_per_image=100000, ignore_index=255):
        super(OHEMPixelSampler, self).__init__()
        # assert
        assert min_kept_per_image > 1
        assert loss_func is None or thresh is None
        # set attrs
        self.loss_func = loss_func
        self.thresh = thresh
        self.min_kept_per_image = min_kept_per_image
        self.ignore_index = ignore_index
    '''sample'''
    @torch.no_grad()
    def sample(self, seg_logits, seg_targets, **kwargs):
        # seg_logits: (N, C, H, W), seg_targets: (N, H, W)
        assert seg_logits.shape[-2:] == seg_targets.shape[-2:]
        # prepare
        seg_targets = seg_targets.long()
        batch_kept = self.min_kept_per_image * seg_targets.size(0)
        valid_mask = (seg_targets != self.ignore_index)
        seg_weights = seg_logits.new_zeros(size=seg_targets.size())
        valid_seg_weights = seg_weights[valid_mask]
        # sample pixels
        if self.thresh is not None:
            seg_probs = F.softmax(seg_logits, dim=1)
            tmp_seg_targets = seg_targets.clone().unsqueeze(1)
            tmp_seg_targets[tmp_seg_targets == self.ignore_index] = 0
            seg_probs = seg_probs.gather(1, tmp_seg_targets).squeeze(1)
            sort_prob, sort_indices = seg_probs[valid_mask].sort()
            if sort_prob.numel() > 0:
                min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)]
            else:
                min_threshold = 0.0
            threshold = max(min_threshold, self.thresh)
            valid_seg_weights[seg_probs[valid_mask] < threshold] = 1.
        else:
            losses = self.loss_func(seg_logits, seg_targets)
            _, sort_indices = losses[valid_mask].sort(descending=True)
            valid_seg_weights[sort_indices[:batch_kept]] = 1.
        # seg_weights: (N, H, W)
        seg_weights[valid_mask] = valid_seg_weights
        # extract sampled pixels
        with torch.enable_grad():
            point_logits = seg_logits.permute(0, 2, 3, 1).contiguous()[seg_weights > 0]
        point_targets = seg_targets[seg_weights > 0]
        # return
        return point_logits, point_targets