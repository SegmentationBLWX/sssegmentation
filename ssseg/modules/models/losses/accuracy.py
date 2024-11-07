'''
Function:
    Implementation of Accuracy
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .misc import bchw2nc


'''calculateaccuracy'''
def calculateaccuracy(x_src, x_tgt, topk=1, thresh=None, ignore_index=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk, return_single = (topk, ), True
    else:
        return_single = False
    maxk = max(topk)
    # if empty
    if x_src.size(0) == 0:
        acc = [x_tgt.new_tensor(0.) for _ in range(len(topk))]
        return acc[0] if return_single else acc
    # assert valid
    assert x_src.ndim == x_tgt.ndim + 1
    assert x_src.size(0) == x_tgt.size(0)
    assert maxk <= x_src.size(1), f'maxk {maxk} exceeds x_src dimension {x_src.size(1)}'
    # topk
    pred_value, pred_label = x_src.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    # num correct
    correct = pred_label.eq(x_tgt.unsqueeze(0).expand_as(pred_label))
    # only x_src values larger than thresh are counted as correct
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    # consider ignore_index
    if ignore_index is not None:
        correct = correct[:, x_tgt != ignore_index]
    # return result
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = x_tgt[x_tgt != ignore_index].numel() + eps
        else:
            total_num = x_tgt.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


'''Accuracy'''
class Accuracy(nn.Module):
    def __init__(self, topk=(1, ), thresh=None, ignore_index=None):
        super(Accuracy, self).__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index
    '''forward'''
    def forward(self, x_src, x_tgt):
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1)
        return calculateaccuracy(x_src, x_tgt, self.topk, self.thresh, self.ignore_index)


'''unittest'''
if __name__ == '__main__':
    x_src = torch.randn(2, 21, 128, 128)
    x_tgt = torch.randint(0, 21, (2, 128, 128))
    # topk
    print(Accuracy(topk=(1,))(x_src, x_tgt))
    print(Accuracy(topk=(1, 2,))(x_src, x_tgt))
    # thresh
    print(Accuracy(topk=(1,), thresh=0.1)(x_src, x_tgt))
    print(Accuracy(topk=(1,), thresh=0.2)(x_src, x_tgt))
    # ignore index
    print(Accuracy(topk=(1,), ignore_index=10)(x_src, x_tgt))
    print(Accuracy(topk=(1,), ignore_index=30)(x_src, x_tgt))