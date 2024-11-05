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
def diceloss(x_src, x_tgt, weight=None, eps=1e-3, reduction='mean', naive_dice=False, ignore_index=-100):
    # convert x_src and x_tgt
    num_classes = x_src.shape[1]
    if (x_src.shape == x_tgt.shape):
        x_src = x_src[:, torch.arange(num_classes) != ignore_index, ...]
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index, ...]
        ignore_index_mask = None
    else:
        ignore_index_mask = (x_tgt != ignore_index).unsqueeze(1).float()
        x_tgt = torch.clamp(x_tgt, min=0, max=num_classes)
        if x_tgt.dim() == 1:
            x_tgt = F.one_hot(x_tgt, num_classes+1).float()
            if num_classes == 1:
                x_tgt = x_tgt[..., 1].unsqueeze(-1)
            else:
                x_tgt = x_tgt[..., :num_classes]
        else:
            x_tgt = F.one_hot(x_tgt, num_classes+1).float()
            if num_classes == 1:
                x_tgt = x_tgt[..., 1].unsqueeze(-1).permute(0, -1, *range(1, x_tgt.dim() - 1)).contiguous()
            else:
                x_tgt = x_tgt[..., :num_classes].permute(0, -1, *range(1, x_tgt.dim() - 1)).contiguous()
    # calculate dice loss
    if ignore_index_mask is not None:
        x_src = x_src * ignore_index_mask
        x_tgt = x_tgt * ignore_index_mask
    x_src = x_src.flatten(1)
    x_tgt = x_tgt.flatten(1)
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
    # reduce loss with weight
    loss = reducelosswithweight(loss, weight, reduction, None)
    # return calculated loss
    return loss


'''DiceLoss'''
class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, activate=True, reduction='mean', naive_dice=False, eps=1e-3, scale_factor=1.0, ignore_index=-100, lowest_loss_value=None):
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
        # calculate loss
        if self.activate:
            if self.use_sigmoid:
                x_src = x_src.sigmoid()
            elif x_src.shape[1] > 1:
                x_src = x_src.softmax(dim=1)
        loss = diceloss(x_src, x_tgt, weight=weight, eps=self.eps, reduction=self.reduction, naive_dice=self.naive_dice, ignore_index=self.ignore_index)
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
            # ignore index
            print('*** TEST on CUDA and CPU with ignore index ***')
            batch_size, h, w = 4, 32, 32
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src, x_tgt))
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src, x_tgt_onehot))
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src.cuda(), x_tgt.cuda()))
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src, x_tgt))
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src, x_tgt_onehot))
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src.cuda(), x_tgt.cuda()))
            print(DiceLoss(reduction='mean', ignore_index=2)(x_src.cuda(), x_tgt_onehot.cuda()))