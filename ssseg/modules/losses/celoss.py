'''
Function:
    define the cross entropy loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F


'''define all'''
__all__ = ['CrossEntropyLoss', 'BinaryCrossEntropyLoss']


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
        'weight': kwargs.get('weight', None),
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


'''
Function:
    binary cross entropy loss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def BinaryCrossEntropyLoss(prediction, target, scale_factor=1.0, **kwargs):
    # expand onehot labels to match the size of prediction
    if prediction.dim() != target.dim():
        assert (prediction.dim() == 2 and target.dim() == 1) or (prediction.dim() == 4 and target.dim() == 3)
        ignore_index = kwargs.get('ignore_index', 255)
        target_binary = target.new_zeros(prediction.shape).type_as(prediction)
        valid_mask = (target >= 0) & (target != ignore_index)
        idxs = torch.nonzero(valid_mask, as_tuple=True)
        if idxs[0].numel() > 0:
            if target.dim() == 3:
                target_binary[idxs[0], target[valid_mask].long(), idxs[1], idxs[2]] = 1
            else:
                target_binary[idxs[0], target[valid_mask].long()] = 1
        prediction = prediction[valid_mask]
        target_binary = target_binary[valid_mask]
        if 'weight' in kwargs: kwargs['weight'] = kwargs['weight'][valid_mask]
    else:
        target_binary = target
    # calculate the loss
    ce_args = {
        'weight': kwargs.get('weight', None),
        'reduction': kwargs.get('reduction', 'mean'),
        'pos_weight': kwargs.get('pos_weight', None)
    }
    loss = F.binary_cross_entropy_with_logits(prediction, target_binary.float(), **ce_args)
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    lowest_loss_value = kwargs.get('lowest_loss_value', None)
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss