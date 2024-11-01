'''
Function:
    Implementation of L1Loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''L1Loss'''
class L1Loss(nn.Module):
    def __init__(self, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt):
        # assert
        assert x_src.size() == x_tgt.size()
        # fetch attributes
        scale_factor, reduction, lowest_loss_value = self.scale_factor, self.reduction, self.lowest_loss_value
        # calculate loss
        loss = F.l1_loss(x_src, x_tgt, reduction=reduction)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss