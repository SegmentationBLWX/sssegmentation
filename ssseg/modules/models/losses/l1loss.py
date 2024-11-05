'''
Function:
    Implementation of L1Loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import reducelosswithweight


'''L1Loss'''
class L1Loss(nn.Module):
    def __init__(self, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert x_src.shape == x_tgt.shape, 'invalid shape of x_src or x_tgt'
        # calculate loss
        loss = F.l1_loss(x_src, x_tgt, reduction='none')
        # reduce loss with weight
        loss = reducelosswithweight(loss, weight, self.reduction, None)
        # rescale loss
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        # return calculated loss
        return loss


'''unittest'''
if __name__ == '__main__':
    for _ in range(3):
        batch_size, c, h, w = 4, 512, 64, 64
        x_src, x_tgt = torch.rand(batch_size, c, h, w), torch.rand(batch_size, c, h, w)
        weight = torch.rand(batch_size, c, h, w)
        # cuda or cpu
        print('*** TEST on CUDA and CPU ***')
        print(L1Loss(reduction='mean')(x_src, x_tgt))
        print(F.l1_loss(x_src, x_tgt, reduction='mean'))
        print(L1Loss(reduction='mean')(x_src.cuda(), x_tgt.cuda()))
        print(F.l1_loss(x_src.cuda(), x_tgt.cuda(), reduction='mean'))
        # weight
        print('*** TEST on CUDA and CPU with weight ***')
        print(L1Loss(reduction='mean')(x_src, x_tgt, weight))
        print((F.l1_loss(x_src, x_tgt, reduction='none') * weight).mean())
        print(L1Loss(reduction='mean')(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print((F.l1_loss(x_src.cuda(), x_tgt.cuda(), reduction='none') * weight.cuda()).mean())