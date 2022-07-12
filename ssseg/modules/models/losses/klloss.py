'''
Function:
    Define the Kullback-Leibler divergence loss measure
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Function:
    KLDivLoss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def KLDivLoss(prediction, target, scale_factor=1.0, temperature=1, reduction='mean', log_target=False, lowest_loss_value=None):
    assert prediction.size() == target.size()
    # parse
    src_distribution = nn.LogSoftmax(dim=1)(prediction / temperature)
    tgt_distribution = nn.Softmax(dim=1)(target / temperature)
    kl_args = {
        'reduction': reduction,
        'log_target': log_target,
    }
    try: nn.KLDivLoss(log_target=False)
    except: kl_args.pop('log_target')
    # calculate the loss
    loss = (temperature ** 2) * nn.KLDivLoss(**kl_args)(src_distribution, tgt_distribution)
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss