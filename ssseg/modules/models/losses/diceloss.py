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
        assert reduction in ['mean', 'sum', 'none']
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt):
        # assert
        assert (len(x_src.shape) == 4 and len(x_tgt.shape == 3)) or (len(x_src.shape) == 2 and len(x_tgt.shape == 1))
        # fetch attributes
        smooth, exponent, reduction, ignore_index = self.smooth, self.exponent, self.reduction, self.ignore_index
        class_weight, scale_factor, lowest_loss_value = self.class_weight, self.scale_factor, self.lowest_loss_value
        # construct loss_cfg
        if self.class_weight is not None:
            class_weight = x_src.new_tensor(self.class_weight)
        else:
            class_weight = None
        dice_cfg = {'smooth': smooth, 'exponent': exponent, 'reduction': reduction, 'class_weight': class_weight, 'ignore_index': ignore_index}
        # calculate loss
        x_src = F.softmax(x_src, dim=1)
        num_classes = x_src.shape[1]
        x_tgt_onehot = F.one_hot(torch.clamp(x_tgt.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (x_tgt != dice_cfg['ignore_index']).long()
        loss = self.diceloss(x_src, x_tgt_onehot, valid_mask, **dice_cfg)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss
    '''diceloss'''
    def diceloss(self, x_src, x_tgt, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255, reduction='mean'):
        assert x_src.shape[0] == x_tgt.shape[0]
        total_loss = 0
        num_classes = x_src.shape[1]
        for i in range(num_classes):
            if i != ignore_index:
                dice_loss = self.binarydiceloss(x_src[:, i], x_tgt[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent)
                if class_weight is not None: dice_loss *= class_weight[i]
                total_loss += dice_loss
        loss = total_loss / num_classes
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss
    '''binarydiceloss'''
    @staticmethod
    def binarydiceloss(x_src, x_tgt, valid_mask, smooth=1, exponent=2):
        assert x_src.shape[0] == x_tgt.shape[0]
        x_src = x_src.reshape(x_src.shape[0], -1)
        x_tgt = x_tgt.reshape(x_tgt.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(x_src, x_tgt) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum(x_src.pow(exponent) + x_tgt.pow(exponent), dim=1) + smooth
        return 1 - num / den