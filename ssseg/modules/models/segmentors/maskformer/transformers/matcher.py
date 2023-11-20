'''
Function:
    Implementation of HungarianMatcher
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


'''HungarianMatcher'''
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_mask=1.0, cost_dice=1.0):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
    '''memory efficient forward'''
    @torch.no_grad()
    def memoryefficientforward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        # iterate through batch size
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)
            out_mask = outputs['pred_masks'][b]
            tgt_ids, tgt_mask = targets[b]['labels'], targets[b]['masks'].to(out_mask)
            cost_class = -out_prob[:, tgt_ids]
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode='nearest')
            out_mask, tgt_mask = out_mask.flatten(1), tgt_mask[:, 0].flatten(1)
            cost_mask = self.sigmoidfocalloss(out_mask, tgt_mask)
            cost_dice = self.diceloss(out_mask, tgt_mask)
            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    '''forward'''
    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memoryefficientforward(outputs, targets)
    '''dice loss'''
    def diceloss(self, inputs, targets):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss
    '''sigmoid focal loss'''
    def sigmoidfocalloss(self, inputs, targets, alpha=0.25, gamma=2.0):
        hw = inputs.shape[1]
        prob = inputs.sigmoid()
        focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
        focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
        if alpha >= 0:
            focal_pos = focal_pos * alpha
            focal_neg = focal_neg * (1 - alpha)
        loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, (1 - targets))
        return loss / hw