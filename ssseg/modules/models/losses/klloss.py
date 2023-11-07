'''
Function:
    Implementation of KLDivLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''KLDivLoss'''
class KLDivLoss(nn.Module):
    def __init__(self, scale_factor=1.0, temperature=1, reduction='mean', log_target=False, lowest_loss_value=None):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        assert prediction.size() == target.size()
        reduction, log_target, temperature, scale_factor, lowest_loss_value = self.reduction, self.log_target, self.temperature, self.scale_factor, self.lowest_loss_value
        # construct loss_cfg
        kl_args = {'reduction': reduction}
        if log_target is not None:
            kl_args.update({'log_target': log_target})
        # calculate loss
        src_distribution = nn.LogSoftmax(dim=1)(prediction / temperature)
        tgt_distribution = nn.Softmax(dim=1)(target / temperature)
        loss = (temperature ** 2) * nn.KLDivLoss(**kl_args)(src_distribution, tgt_distribution)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss