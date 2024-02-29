'''
Function:
    Implementation of UGSampler
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F
from .base import BasePixelSampler


'''UGSampler'''
class UGSampler(BasePixelSampler):
    def __init__(self, num_points, oversample_ratio, importance_sample_ratio, ignore_index=255, num_classes=None, reformat_target=True):
        super(UGSampler, self).__init__()
        # assert
        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
        # set attributes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.reformat_target = reformat_target
    '''sample'''
    @torch.no_grad()
    def sample(self, seg_logits, seg_targets, **kwargs):
        # reformat targets if necessary
        if self.reformat_target:
            # (bs, nc, h, w) >>> (bs * nc, h, w) and (bs, h, w) >>> (bs * nc, h, w)
            seg_logits_new, seg_targets_new = [], []
            for bs in range(seg_logits.shape[0]):
                # [(nc, h, w), ...]
                seg_logits_new.append(seg_logits[bs])
                # (h, w)
                seg_targets_per_img = seg_targets[bs]
                # convert to (nc, h, w)
                masks = []
                for label in range(self.num_classes):
                    masks.append((seg_targets_per_img == label).unsqueeze(0))
                masks = torch.cat(masks, dim=0).type_as(seg_logits).long()
                # [(nc, h, w), ...]
                seg_targets_new.append(masks)
            # concat
            seg_logits = torch.cat(seg_logits_new, dim=0)
            seg_targets = torch.cat(seg_targets_new, dim=0)
        # (num_proposals, h, w)
        assert len(seg_logits.shape) == 3 and len(seg_targets.shape) == 3
        with torch.enable_grad():
            seg_logits, seg_targets = seg_logits[:, None], seg_targets[:, None]
        # sample point_coords
        point_coords = self.getuncertainpointcoordswithrandomness(
            seg_logits, lambda logits: self.calculateuncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio,
        )
        # point_logits and point_targets
        point_targets = self.pointsample(seg_targets, point_coords, align_corners=False).squeeze(1)
        with torch.enable_grad():
            point_logits = self.pointsample(seg_logits, point_coords, align_corners=False).squeeze(1)
        # return
        return point_logits, point_targets
    '''calculateuncertainty'''
    @staticmethod
    def calculateuncertainty(logits):
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))
    '''pointsample'''
    @staticmethod
    def pointsample(inputs, point_coords, **kwargs):
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        output = F.grid_sample(inputs, 2.0 * point_coords - 1.0, **kwargs)
        if add_dim:
            output = output.squeeze(3)
        return output
    '''getuncertainpointcoordswithrandomness'''
    @staticmethod
    def getuncertainpointcoordswithrandomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
        # assert
        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
        # get uncertain point coords with randomness
        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = UGSampler.pointsample(coarse_logits, point_coords, align_corners=False)
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        if num_random_points > 0:
            point_coords = torch.cat([point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)], dim=1)
        # return outputs
        return point_coords