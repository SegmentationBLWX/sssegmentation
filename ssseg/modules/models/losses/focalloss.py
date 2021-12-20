'''
Function:
    define the focal loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss


'''define all'''
__all__ = ['SigmoidFocalLoss']


'''
Function:
    sigmoid focal loss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def SigmoidFocalLoss(prediction, target, scale_factor=1.0, **kwargs):
    # filter according to ignore_index
    ignore_index = kwargs.get('ignore_index', None)
    if ignore_index is not None:
        num_classes = prediction.size(-1)
        mask = (target != ignore_index)
        prediction, target = prediction[mask].view(-1, num_classes), target[mask].view(-1)
    # calculate the loss
    gamma, alpha, weight, reduction = kwargs.get('gamma', 2), kwargs.get('alpha', 0.25), kwargs.get('weight', None), kwargs.get('reduction', 'mean')
    loss = sigmoid_focal_loss(prediction, target.long(), gamma, alpha, weight, reduction)
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    lowest_loss_value = kwargs.get('lowest_loss_value', None)
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss