'''
Function:
    Implementation of CrossEntropyLoss and BinaryCrossEntropyLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''CrossEntropyLoss'''
class CrossEntropyLoss(nn.Module):
    def __init__(self, scale_factor=1.0, weight=None, ignore_index=255, reduction='mean', lowest_loss_value=None, label_smoothing=None, force_target_as_long=True):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.label_smoothing = label_smoothing
        self.lowest_loss_value = lowest_loss_value
        self.force_target_as_long = force_target_as_long
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        weight, reduction, ignore_index = self.weight, self.reduction, self.ignore_index
        scale_factor, label_smoothing, lowest_loss_value = self.scale_factor, self.label_smoothing, self.lowest_loss_value
        if (weight is not None) and isinstance(weight, torch.Tensor):
            weight = weight.type_as(prediction)
        # construct loss_cfg
        ce_args = {'weight': weight, 'ignore_index': ignore_index, 'reduction': reduction}
        if label_smoothing is not None:
            ce_args.update({'label_smoothing': label_smoothing})
        # calculate loss
        if self.force_target_as_long:
            loss = F.cross_entropy(prediction, target.long(), **ce_args)
        else:
            loss = F.cross_entropy(prediction, target, **ce_args)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss


'''BinaryCrossEntropyLoss'''
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, scale_factor=1.0, weight=None, ignore_index=255, reduction='mean', pos_weight=None, lowest_loss_value=None):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        weight, reduction, ignore_index = self.weight, self.reduction, self.ignore_index
        scale_factor, pos_weight, lowest_loss_value = self.scale_factor, self.pos_weight, self.lowest_loss_value
        if (weight is not None) and isinstance(weight, torch.Tensor):
            weight = weight.type_as(prediction)
        if (pos_weight is not None) and isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.type_as(prediction)
        # expand onehot labels to match the size of prediction
        if prediction.dim() != target.dim():
            assert (prediction.dim() == 2 and target.dim() == 1) or (prediction.dim() == 4 and target.dim() == 3)
            if prediction.dim() == 4:
                prediction = prediction.permute(0, 2, 3, 1).contiguous()
                prediction = prediction.reshape(-1, prediction.shape[-1])
                target = target.reshape(-1)
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
            if weight:
                weight = weight[valid_mask]
        else:
            target_binary = target
        # construct loss_cfg
        ce_args = {'weight': weight, 'reduction': reduction, 'pos_weight': pos_weight}
        # calculate loss
        loss = F.binary_cross_entropy_with_logits(prediction, target_binary.float(), **ce_args)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss