'''
Function:
    define the cross entropy loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F


'''define all'''
__all__ = ['CrossEntropyLoss']


'''
Function:
    cross entropy loss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def CrossEntropyLoss(prediction, target, scale_factor=1.0, **kwargs):
    # calculate the loss
    ce_args = {
        'weight': kwargs.get('cls_weight', None),
        'ignore_index': kwargs.get('ignore_index', 255),
        'reduction': kwargs.get('reduction', 'mean'),
    }
    loss = F.cross_entropy(prediction, target.long(), **ce_args)
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    lowest_loss_value = kwargs.get('lowest_loss_value', None)
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss