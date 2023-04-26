'''
Function:
    Implementation of calculateaccuracy
Author:
    Zhenchao Jin
'''
import torch


'''calculateaccuracy'''
def calculateaccuracy(prediction, target, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk, return_single = (topk, ), True
    else:
        return_single = False
    maxk = max(topk)
    # if empty
    if prediction.size(0) == 0:
        acc = [prediction.new_tensor(0.) for i in range(len(topk))]
        return acc[0] if return_single else acc
    # assert valid
    assert prediction.ndim == target.ndim + 1
    assert prediction.size(0) == target.size(0)
    assert maxk <= prediction.size(1), f'maxk {maxk} exceeds prediction dimension {prediction.size(1)}'
    # topk
    pred_value, pred_label = prediction.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    # num correct
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    # Only prediction values larger than thresh are counted as correct
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    # return result
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res[0] if return_single else res