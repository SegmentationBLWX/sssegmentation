'''
Function:
    Implementation of KLDivLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''KLDivLoss'''
class KLDivLoss(nn.Module):
    def __init__(self, scale_factor=1.0, temperature=1, reduction='mean', lowest_loss_value=None):
        super(KLDivLoss, self).__init__()
        assert reduction in ['batchmean', 'mean', 'sum', 'none']
        self.reduction = reduction
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert x_src.shape == x_tgt.shape, 'invalid shape of x_src or x_tgt'
        # calculate loss
        x_src = F.log_softmax(x_src / self.temperature, dim=1)
        x_tgt = F.softmax(x_tgt / self.temperature, dim=1)
        loss = F.kl_div(x_src, x_tgt, reduction='none', log_target=False)
        loss = loss * self.temperature**2
        # reduce loss with weight
        if weight is not None:
            loss = loss * weight
        batch_size = x_src.shape[0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'batchmean':
            loss = loss.sum() / batch_size
        elif self.reduction == 'sum':
            loss = loss.sum()
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
        x_src, x_tgt = torch.randn(batch_size, c, h, w), torch.randn(batch_size, c, h, w)
        weight = torch.rand(batch_size, c, h, w)
        # cuda or cpu
        print('*** TEST on CUDA and CPU ***')
        print(KLDivLoss(reduction='mean')(x_src, x_tgt))
        print(F.kl_div(x_src.log_softmax(dim=1), x_tgt.softmax(dim=1), reduction='mean'))
        print(KLDivLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda()))
        print(F.kl_div(x_src.cuda().log_softmax(dim=1), x_tgt.cuda().softmax(dim=1), reduction='mean'))
        # weight
        print('*** TEST on CUDA and CPU with weight ***')
        print(KLDivLoss(reduction='mean')(x_src, x_tgt, weight))
        print((F.kl_div(x_src.log_softmax(dim=1), x_tgt.softmax(dim=1), reduction='none') * weight).mean())
        print(KLDivLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print((F.kl_div(x_src.cuda().log_softmax(dim=1), x_tgt.cuda().softmax(dim=1), reduction='none') * weight.cuda()).mean())