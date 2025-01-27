'''
Function:
    Implementation of Utils related loss functions
Author:
    Zhenchao Jin
'''
import torch


'''reduceloss'''
def reduceloss(loss, reduction='mean', avg_factor=None):
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        if avg_factor is None:
            return torch.mean(loss)
        return torch.sum(loss) / avg_factor
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss


'''reducelosswithweight'''
def reducelosswithweight(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight.float()
    return reduceloss(loss=loss, reduction=reduction, avg_factor=avg_factor)


'''bchw2nc'''
def bchw2nc(tensor: torch.Tensor):
    tensor = tensor.transpose(0, 1)
    tensor = tensor.reshape(tensor.size(0), -1)
    tensor = tensor.transpose(0, 1).contiguous()
    return tensor