'''
Function:
    Implementation of CosineSimilarityLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''CosineSimilarityLoss'''
class CosineSimilarityLoss(nn.Module):
    def __init__(self, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
        super(CosineSimilarityLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
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
        loss = 1 - F.cosine_similarity(x_src, x_tgt, dim=1)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum': 
            loss = loss.sum()
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss