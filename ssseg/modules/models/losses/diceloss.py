'''
Function:
    Define the dice loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F


'''
Function:
    DiceLoss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def DiceLoss(prediction, target, scale_factor=1.0, smooth=1, exponent=2, reduction='mean', class_weight=None, ignore_index=255, lowest_loss_value=None):
    '''binary dice loss'''
    def BinaryDiceLoss(pred, target, valid_mask, smooth=1, exponent=2):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
        return 1 - num / den
    '''unwrapped dice loss'''
    def _DiceLoss(pred, target, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255):
        assert pred.shape[0] == target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != ignore_index:
                dice_loss = BinaryDiceLoss(pred[:, i], target[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent)
                if class_weight is not None: dice_loss *= class_weight[i]
                total_loss += dice_loss
        return total_loss / num_classes
    # calculate the loss
    dice_cfg = {
        'smooth': smooth,
        'exponent': exponent,
        'reduction': reduction,
        'class_weight': class_weight,
        'ignore_index': ignore_index,
    }
    if dice_cfg['class_weight'] is not None:
        class_weight = prediction.new_tensor(dice_cfg['class_weight'])
    else:
        class_weight = None
    prediction = F.softmax(prediction, dim=1)
    num_classes = prediction.shape[1]
    one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
    valid_mask = (target != dice_cfg['ignore_index']).long()
    loss = _DiceLoss(prediction, one_hot_target, valid_mask, **dice_cfg)
    if dice_cfg['reduction'] == 'mean':
        loss = loss.mean()
    elif dice_cfg['reduction'] == 'sum':
        loss = loss.sum()
    else:
        assert dice_cfg['reduction'] == 'none', 'only support reduction in [mean, sum, none]'
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss