'''
Function:
    Implementation of BasePixelSampler (here, we implement a RandomPixelSampler as BasePixelSampler)
Author:
    Zhenchao Jin
'''
import random


'''BasePixelSampler'''
class BasePixelSampler(object):
    def __init__(self, sample_ratio=0.5):
        self.sample_ratio = sample_ratio
    '''sample'''
    def sample(self, seg_logits, seg_targets, **kwargs):
        # seg_logits: (N, C, H, W), seg_targets: (N, H, W)
        assert seg_logits.shape[-2:] == seg_targets.shape[-2:]
        n, c, h, w = seg_logits.shape
        # num pixels
        num_pixels = h * w
        sampled_num_pixels = int(self.sample_ratio * num_pixels)
        # indices
        indices = list(range(num_pixels))
        random.shuffle(indices)
        indices = indices[:sampled_num_pixels]
        # select
        seg_logits = seg_logits.permute(2, 3, 0, 1).contiguous().reshape(h * w, n, c)
        seg_logits = seg_logits[indices].permute(1, 2, 0).contiguous().reshape(n * c, sampled_num_pixels)
        seg_targets = seg_targets.permute(1, 2, 0).contiguous().reshape(h * w, n)
        seg_targets = seg_targets[indices].permute(1, 0).contiguous().reshape(n, sampled_num_pixels)
        # return
        return seg_logits, seg_targets