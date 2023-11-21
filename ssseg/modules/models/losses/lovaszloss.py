'''
Function:
    Implementation of LovaszLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''LovaszGrad: computes gradient of the lovasz extension w.r.t sorted errors'''
def LovaszGrad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: jaccard[1: p] = jaccard[1: p] - jaccard[0: -1]
    return jaccard


'''FlattenBinaryLogits: flattens predictions in the batch (binary case), remove labels equal to ignore_index.'''
def FlattenBinaryLogits(logits, labels, ignore_index=None):
    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None: return logits, labels
    valid = (labels != ignore_index)
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


'''FlattenProbs: flattens predictions in the batch'''
def FlattenProbs(probs, labels, ignore_index=None):
    if probs.dim() == 3:
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is None: return probs, labels
    valid = (labels != ignore_index)
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


'''LovaszHingeFlat'''
def LovaszHingeFlat(logits, labels):
    if len(labels) == 0: return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = LovaszGrad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


'''LovaszHingeLoss'''
def LovaszHingeLoss(prediction, target, scale_factor=1.0, per_image=False, reduction='mean', ignore_index=255, lowest_loss_value=None):
    # calculate the loss
    lovasz_cfg = {'per_image': per_image, 'reduction': reduction, 'ignore_index': ignore_index}
    if lovasz_cfg['per_image']:
        loss = [LovaszHingeFlat(*FlattenBinaryLogits(logit.unsqueeze(0), label.unsqueeze(0), lovasz_cfg['ignore_index'])) for logit, label in zip(prediction, target)]
        loss = torch.stack(loss)
    else:
        loss = LovaszHingeFlat(*FlattenBinaryLogits(prediction, target, lovasz_cfg['ignore_index']))
    if lovasz_cfg['reduction'] == 'mean':
        loss = loss.mean()
    elif lovasz_cfg['reduction'] == 'sum':
        loss = loss.sum()
    else:
        assert lovasz_cfg['reduction'] == 'none', 'only support reduction in [mean, sum, none]'
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss


'''LovaszSoftmaxFlat'''
def LovaszSoftmaxFlat(probs, labels, classes='present', class_weight=None):
    if probs.numel() == 0: return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0): continue
        if C == 1:
            if len(classes) > 1: raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, LovaszGrad(fg_sorted))
        if class_weight is not None: loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


'''LovaszSoftmaxLoss'''
def LovaszSoftmaxLoss(prediction, target, scale_factor=1.0, per_image=False, classes='present', reduction='mean', ignore_index=255, class_weight=None, lowest_loss_value=None):
    # calculate the loss
    prediction = F.softmax(prediction, dim=1)
    if class_weight is not None:
        class_weight = class_weight.type_as(prediction)
    lovasz_cfg = {'per_image': per_image, 'classes': classes, 'reduction': reduction, 'ignore_index': ignore_index, 'class_weight': class_weight}
    if lovasz_cfg['per_image']:
        loss = [LovaszSoftmaxFlat(*FlattenProbs(prob.unsqueeze(0), label.unsqueeze(0), lovasz_cfg['ignore_index']), classes=lovasz_cfg['classes'], class_weight=lovasz_cfg['class_weight']) for prob, label in zip(prediction, target)]
        loss = torch.stack(loss)
    else:
        loss = LovaszSoftmaxFlat(*FlattenProbs(prediction, target, lovasz_cfg['ignore_index']), classes=lovasz_cfg['classes'], class_weight=lovasz_cfg['class_weight'])
    if lovasz_cfg['reduction'] == 'mean':
        loss = loss.mean()
    elif lovasz_cfg['reduction'] == 'sum':
        loss = loss.sum()
    else:
        assert lovasz_cfg['reduction'] == 'none', 'only support reduction in [mean, sum, none]'
    # scale the loss
    loss = loss * scale_factor
    # return the final loss
    if lowest_loss_value:
        return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    return loss


'''LovaszLoss'''
class LovaszLoss(nn.Module):
    def __init__(self, mode='multi_class', **kwargs):
        super(LovaszLoss, self).__init__()
        self.mode = mode
        self.loss_args = kwargs
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        mode = self.mode
        # supported modes
        supported_modes = {
            'binary': LovaszHingeLoss, 'multi_class': LovaszSoftmaxLoss,
        }
        # construct loss_cfg
        lovasz_args = self.loss_args.copy()
        lovasz_args.update({'prediction': prediction, 'target': target})
        # calculate loss
        loss = supported_modes[mode](**lovasz_args)
        # return
        return loss