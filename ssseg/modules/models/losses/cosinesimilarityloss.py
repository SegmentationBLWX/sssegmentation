'''
Function:
    Define the cosine similarity loss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Function:
    CosineSimilarityLoss
Arguments:
    --prediction: prediction of the network
    --target: ground truth
    --scale_factor: scale the loss for loss balance
    --lowest_loss_value: added inspired by ICML2020, "Do We Need Zero Training Loss After Achieving Zero Training Error", https://arxiv.org/pdf/2002.08709.pdf
'''
def CosineSimilarityLoss(prediction, target, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
    # calculate the loss
    assert prediction.shape == target.shape
    loss = 1 - F.cosine_similarity(prediction, target, dim=1)
    if reduction == 'mean': loss = loss.mean()
    elif reduction == 'sum': loss = loss.sum()
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss