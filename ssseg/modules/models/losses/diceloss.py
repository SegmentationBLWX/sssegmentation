'''
Function:
    Implementation of DiceLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''DiceLoss'''
class DiceLoss(nn.Module):
    def __init__(self, scale_factor=1.0, smooth=1, exponent=2, reduction='mean', class_weight=None, ignore_index=255, lowest_loss_value=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        smooth, exponent, reduction, ignore_index = self.smooth, self.exponent, self.reduction, self.ignore_index
        class_weight, scale_factor, lowest_loss_value = self.class_weight, self.scale_factor, self.lowest_loss_value
        # construct loss_cfg
        if self.class_weight is not None:
            class_weight = prediction.new_tensor(self.class_weight)
        else:
            class_weight = None
        dice_cfg = {
            'smooth': smooth, 'exponent': exponent, 'reduction': reduction, 'class_weight': class_weight, 'ignore_index': ignore_index,
        }
        # calculate loss
        prediction = F.softmax(prediction, dim=1)
        num_classes = prediction.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (target != dice_cfg['ignore_index']).long()
        loss = self.diceloss(prediction, one_hot_target, valid_mask, **dice_cfg)
        if dice_cfg['reduction'] == 'mean':
            loss = loss.mean()
        elif dice_cfg['reduction'] == 'sum':
            loss = loss.sum()
        else:
            assert dice_cfg['reduction'] == 'none', 'only support reduction in [mean, sum, none]'
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss
    '''diceloss'''
    def diceloss(self, pred, target, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255):
        assert pred.shape[0] == target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != ignore_index:
                dice_loss = self.binarydiceloss(pred[:, i], target[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent)
                if class_weight is not None: dice_loss *= class_weight[i]
                total_loss += dice_loss
        return total_loss / num_classes
    '''binarydiceloss'''
    @staticmethod
    def binarydiceloss(pred, target, valid_mask, smooth=1, exponent=2):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
        return 1 - num / den