'''
Function:
    Implementation of SigmoidFocalLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
try:
    from mmcv.ops import sigmoid_focal_loss
except:
    sigmoid_focal_loss = None


'''SigmoidFocalLoss'''
class SigmoidFocalLoss(nn.Module):
    def __init__(self, scale_factor=1.0, gamma=2, alpha=0.25, weight=None, reduction='mean', ignore_index=None, lowest_loss_value=None):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        alpha, gamma, weight, lowest_loss_value = self.alpha, self.gamma, self.weight, self.lowest_loss_value
        scale_factor, reduction, ignore_index = self.scale_factor, self.reduction, self.ignore_index
        # filter according to ignore_index
        if ignore_index is not None:
            num_classes = prediction.size(-1)
            mask = (target != ignore_index)
            prediction, target = prediction[mask].view(-1, num_classes), target[mask].view(-1)
        # calculate loss
        loss = sigmoid_focal_loss(prediction, target.long(), gamma, alpha, weight, reduction)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss