'''
Function:
    Define the l1 loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F


'''
Function:
    L1Loss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def L1Loss(prediction, target, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
    # calculate the loss
    loss = F.l1_loss(prediction, target, reduction=reduction)
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss