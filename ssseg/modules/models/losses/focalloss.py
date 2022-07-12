'''
Function:
    Define the focal loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss


'''
Function:
    SigmoidFocalLoss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def SigmoidFocalLoss(prediction, target, scale_factor=1.0, gamma=2, alpha=0.25, weight=None, reduction='mean', ignore_index=None, lowest_loss_value=None):
    # filter according to ignore_index
    if ignore_index is not None:
        num_classes = prediction.size(-1)
        mask = (target != ignore_index)
        prediction, target = prediction[mask].view(-1, num_classes), target[mask].view(-1)
    # calculate the loss
    loss = sigmoid_focal_loss(prediction, target.long(), gamma, alpha, weight, reduction)
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss