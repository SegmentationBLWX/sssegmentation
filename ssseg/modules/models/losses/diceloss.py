'''
Function:
    Implementation of DiceLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import reducelosswithweight


'''diceloss'''
def diceloss(x_src, x_tgt, weight=None, eps=1e-3, reduction='mean', naive_dice=False, valid_mask=None):
    # calculate dice loss
    final_weight = torch.ones(x_src.shape[0]).type_as(x_src)
    x_src = x_src * valid_mask.unsqueeze(1).float()
    x_tgt = x_tgt.float() * valid_mask.unsqueeze(1).float()
    x_src = x_src.flatten(1)
    x_tgt = x_tgt.flatten(1).float()
    a = torch.sum(x_src * x_tgt, 1)
    if naive_dice:
        b = torch.sum(x_src, 1)
        c = torch.sum(x_tgt, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(x_src * x_src, 1) + eps
        c = torch.sum(x_tgt * x_tgt, 1) + eps
        d = (2 * a) / (b + c)
    loss = 1 - d
    # reweighted loss
    if weight is not None:
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    # reduce loss with weight
    if valid_mask.dim() == 1:
        avg_factor = (valid_mask > 0).sum()
    else:
        avg_factor = ((valid_mask > 0).flatten(1).sum(-1) > 0).sum()
    loss = reducelosswithweight(loss, final_weight, reduction, avg_factor)
    # return calculated loss
    return loss


'''DiceLoss'''
class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, activate=True, reduction='mean', naive_dice=False, eps=1e-3, scale_factor=1.0, ignore_index=255, lowest_loss_value=None):
        super(DiceLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.eps = eps
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert (x_src.shape == x_tgt.shape) or (x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:]), 'invalid shape of x_src or x_tgt'
        if weight is None: weight = torch.ones(x_src.shape[0]).type_as(x_src)
        assert weight.dim() == 1 and weight.shape[0] == x_src.shape[0], 'invalid shape of weight'
        # convert x_src and x_tgt
        if (x_src.shape != x_tgt.shape):
            valid_mask = (x_tgt != self.ignore_index)
            num_classes = x_src.shape[1]
            x_tgt = torch.clamp(x_tgt, min=0, max=num_classes)
            if x_tgt.dim() == 1:
                x_tgt = F.one_hot(x_tgt, num_classes+1)
                x_tgt = x_tgt[..., :num_classes]
            else:
                x_tgt = F.one_hot(x_tgt, num_classes+1)
                x_tgt = x_tgt[..., :num_classes].permute(0, -1, *range(1, x_tgt.dim() - 1)).contiguous()
        else:
            valid_mask = (x_tgt.argmax(dim=1) != self.ignore_index)
        if self.activate:
            if self.use_sigmoid:
                x_src = x_src.sigmoid()
            elif x_src.shape[1] > 1:
                x_src = x_src.softmax(dim=1)
        # calculate loss
        loss = diceloss(x_src, x_tgt, weight=weight, eps=self.eps, reduction=self.reduction, naive_dice=self.naive_dice, valid_mask=valid_mask)
        # rescale loss
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        # return calculated loss
        return loss


'''unittest'''
if __name__ == '__main__':
    for num_classes in [2, 21, 151]:
        with torch.no_grad():
            # cuda or cpu
            print('*** TEST on CUDA and CPU ***')
            batch_size, h, w = 4, 32, 32
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(DiceLoss(reduction='mean')(x_src, x_tgt))
            print(DiceLoss(reduction='mean')(x_src, x_tgt_onehot))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda()))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(DiceLoss(reduction='mean')(x_src, x_tgt))
            print(DiceLoss(reduction='mean')(x_src, x_tgt_onehot))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda()))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda()))
            # weight
            print('*** TEST on CUDA and CPU with weight ***')
            batch_size, h, w = 4, 32, 32
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            weight = torch.rand(x_src.shape[0])
            print(DiceLoss(reduction='mean')(x_src, x_tgt, weight))
            print(DiceLoss(reduction='mean')(x_src, x_tgt_onehot, weight))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            weight = torch.rand(x_src.shape[0])
            print(DiceLoss(reduction='mean')(x_src, x_tgt, weight))
            print(DiceLoss(reduction='mean')(x_src, x_tgt_onehot, weight))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
            print(DiceLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
            # naive dice
            print('*** TEST on CUDA and CPU with naive dice ***')
            batch_size, h, w = 4, 32, 32
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src, x_tgt))
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src, x_tgt_onehot))
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src.cuda(), x_tgt.cuda()))
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src, x_tgt))
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src, x_tgt_onehot))
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src.cuda(), x_tgt.cuda()))
            print(DiceLoss(reduction='mean', naive_dice=True)(x_src.cuda(), x_tgt_onehot.cuda()))