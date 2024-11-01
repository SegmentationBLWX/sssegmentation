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
    def __init__(self, scale_factor=1.0, weight=None, ignore_index=255, reduction='mean', lowest_loss_value=None, label_smoothing=None):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.label_smoothing = label_smoothing
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt):
        # assert
        assert (len(x_src.shape) == 4 and len(x_tgt.shape == 3)) or (len(x_src.shape) == 2 and len(x_tgt.shape == 1)) or (x_src.size() == x_tgt.size())
        # fetch attributes
        weight, reduction, ignore_index = self.weight, self.reduction, self.ignore_index
        scale_factor, label_smoothing, lowest_loss_value = self.scale_factor, self.label_smoothing, self.lowest_loss_value
        if (weight is not None) and isinstance(weight, torch.Tensor):
            weight = weight.type_as(x_src)
        # construct loss_cfg
        ce_args = {'weight': weight, 'ignore_index': ignore_index, 'reduction': reduction}
        if label_smoothing is not None:
            ce_args.update({'label_smoothing': label_smoothing})
        # calculate loss
        loss = F.cross_entropy(x_src, x_tgt, **ce_args)
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
    def forward(self, x_src, x_tgt):
        # assert
        assert (len(x_src.shape) == 4 and len(x_tgt.shape == 3)) or (len(x_src.shape) == 2 and len(x_tgt.shape == 1)) or (x_src.size() == x_tgt.size())
        # convert x_src and x_tgt
        if x_src.dim() == 4:
            num_classes = x_src.size(1)
            x_src = x_src.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
            x_tgt = x_tgt.reshape(-1)
        # fetch attributes
        weight, reduction, ignore_index = self.weight, self.reduction, self.ignore_index
        scale_factor, pos_weight, lowest_loss_value = self.scale_factor, self.pos_weight, self.lowest_loss_value
        if (weight is not None) and isinstance(weight, torch.Tensor):
            weight = weight.type_as(x_src)
        if (pos_weight is not None) and isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.type_as(x_src)
        # filter ignore_index
        valid_mask = (x_tgt >= 0) & (x_tgt != ignore_index)
        x_src, x_tgt = x_src[valid_mask], x_tgt[valid_mask]
        if weight:
            weight = weight[valid_mask]
        # expand onehot labels to match the size of x_src
        if x_src.dim() != x_tgt.dim():
            num_classes = x_src.size(1)
            x_tgt = F.one_hot(torch.clamp(x_tgt.long(), 0, num_classes - 1), num_classes=num_classes)
        # construct loss_cfg
        ce_args = {'weight': weight, 'reduction': reduction, 'pos_weight': pos_weight}
        # calculate loss
        loss = F.binary_cross_entropy_with_logits(x_src, x_tgt.float(), **ce_args)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss