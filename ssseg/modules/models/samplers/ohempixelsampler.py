'''
Function:
    OHEM pixel sampler
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
        assert min_kept_per_image > 1
        # set attrs
        self.loss_func = loss_func
        self.thresh = thresh
        self.min_kept_per_image = min_kept_per_image
        self.ignore_index = ignore_index
    '''sample pixels that have high loss or with low prediction confidence'''
    def sample(self, seg_logit, seg_label):
        # seg_logit: (N, C, H, W), seg_label: (N, 1, H, W)
        assert seg_logit.shape[2:] == seg_label.shape[2:]
        assert seg_label.shape[1] == 1
        # sample pixels
        with torch.no_grad():
            seg_label = seg_label.squeeze(1).long()
            batch_kept = self.min_kept_per_image * seg_label.size(0)
            valid_mask = (seg_label != self.ignore_index)
            seg_weight = seg_logit.new_zeros(size=seg_label.size())
            valid_seg_weight = seg_weight[valid_mask]
            if self.thresh is not None:
                seg_prob = F.softmax(seg_logit, dim=1)
                tmp_seg_label = seg_label.clone().unsqueeze(1)
                tmp_seg_label[tmp_seg_label == self.ignore_index] = 0
                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
                sort_prob, sort_indices = seg_prob[valid_mask].sort()
                if sort_prob.numel() > 0:
                    min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0
                threshold = max(min_threshold, self.thresh)
                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
            else:
                losses = self.loss_func(
                    seg_logit,
                    seg_label,
                )
                _, sort_indices = losses[valid_mask].sort(descending=True)
                valid_seg_weight[sort_indices[:batch_kept]] = 1.
        seg_weight[valid_mask] = valid_seg_weight
        # seg_weight: (N, H, W)
        return seg_weight