'''
Function:
    Implementation of HungarianMatcher
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....samplers.ugsampler import UGSampler
from scipy.optimize import linear_sum_assignment


'''batchdiceloss'''
def batchdiceloss(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


'''batchsigmoidceloss'''
def batchsigmoidceloss(inputs, targets):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))
    return loss / hw


'''HungarianMatcher'''
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1, num_points=0):
        super(HungarianMatcher, self).__init__()
        # assert 
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'
        # set attributes
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
    '''memoryefficientforward'''
    @torch.no_grad()
    def memoryefficientforward(self, outputs, targets):
        from torch.cuda.amp import autocast
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        # iterate through batch size
        for b in range(bs):
            # [num_queries, num_classes]
            out_prob = outputs['pred_logits'][b].softmax(-1)
            tgt_ids = targets[b]['labels']
            # compute the classification cost
            cost_class = -out_prob[:, tgt_ids]
            # [num_queries, H_pred, W_pred]
            out_mask = outputs['pred_masks'][b]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]['masks'].to(out_mask)
            out_mask, tgt_mask = out_mask[:, None], tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = UGSampler.pointsample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_mask = UGSampler.pointsample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            # disable autocast
            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # compute the focal loss between masks
                cost_mask = batchsigmoidceloss(out_mask, tgt_mask)
                # compute the dice loss betwen masks
                cost_dice = batchdiceloss(out_mask, tgt_mask)
            # final cost matrix
            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        # return outputs
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    '''forward'''
    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memoryefficientforward(outputs, targets)