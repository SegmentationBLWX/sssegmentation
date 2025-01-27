'''
Function:
    Implementation of FocalLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import reducelosswithweight, bchw2nc
try:
    from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
    from mmcv.ops import softmax_focal_loss as _softmax_focal_loss
except:
    _sigmoid_focal_loss = None
    _softmax_focal_loss = None


'''pysigmoidfocalloss'''
def pysigmoidfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean', norm_loss_with_class_weight=True):
    # convert x_src and x_tgt to tensor with shape (N, num_classes)
    original_shape, num_classes = x_src.shape, x_src.size(1)
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = (x_tgt != ignore_index)
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes+1).float()
        if num_classes == 1:
            x_tgt = x_tgt[:, 1].unsqueeze(-1)
        else:
            x_tgt = x_tgt[:, :num_classes]
    # convert weight, F.binary_cross_entropy_with_logits always return tensor with shape (N, num_classes)
    if weight is not None:
        weight = bchw2nc(weight).float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    # calculate sigmoid focal loss
    x_src_sigmoid, x_tgt = x_src.sigmoid(), x_tgt.type_as(x_src)
    one_minus_pt = (1 - x_src_sigmoid) * x_tgt + x_src_sigmoid * (1 - x_tgt)
    focal_weight = (alpha * x_tgt + (1 - alpha) * (1 - x_tgt)) * one_minus_pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(x_src, x_tgt, reduction='none') * focal_weight
    # apply class_weight
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        avg_factor = (class_weight * x_tgt).sum()
        if not norm_loss_with_class_weight:
            avg_factor = None
    else:
        avg_factor = None
    # reduce loss with weight
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    # return calculated loss
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


'''sigmoidfocalloss'''
def sigmoidfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean', norm_loss_with_class_weight=True):
    # mmcv.ops._sigmoid_focal_loss only accept x_tgt with class labels, e.g., (B, C, H, W) as inputs and (B, H, W) as targets
    if x_src.shape == x_tgt.shape or x_src.shape[1] == 1:
        return pysigmoidfocalloss(x_src, x_tgt, weight=weight, gamma=gamma, alpha=alpha, class_weight=class_weight, ignore_index=ignore_index, reduction=reduction)
    # convert x_src and x_tgt to tensors with shape (N, num_classes) and (N,)
    original_shape, num_classes = x_src.shape, x_src.size(1)
    x_src = bchw2nc(x_src)
    x_tgt = x_tgt.view(-1).contiguous().long()
    ignore_index_mask = (x_tgt != ignore_index)
    x_src = x_src[ignore_index_mask]
    x_tgt = x_tgt[ignore_index_mask]
    # convert weight, mmcv.ops._sigmoid_focal_loss always return tensor with shape (N, num_classes)
    if weight is not None:
        weight = bchw2nc(weight).float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    # calculate sigmoid focal loss
    loss = _sigmoid_focal_loss(x_src, x_tgt, gamma, alpha, None, 'none')
    # apply class_weight
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes+1).float()
        x_tgt = x_tgt[:, :num_classes]
        avg_factor = (class_weight * x_tgt).sum()
        if not norm_loss_with_class_weight:
            avg_factor = None
    else:
        avg_factor = None
    # reduce loss with weight
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    # return calculated loss
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


'''pysoftmaxfocalloss'''
def pysoftmaxfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean', norm_loss_with_class_weight=True):
    # convert x_src and x_tgt to tensor with shape (N, num_classes)
    original_shape, num_classes = x_src.shape, x_src.size(1)
    assert num_classes > 1, 'num classes of inputs should be larger than 1'
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = (x_tgt != ignore_index)
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes+1).float()
        x_tgt = x_tgt[:, :num_classes]
    # convert weight, F.cross_entropy always return tensor with shape (N,)
    if weight is not None:
        weight = weight.view(-1).contiguous().float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    # calculate softmax focal loss
    loss = F.cross_entropy(x_src, x_tgt.argmax(dim=1), reduction='none')
    loss = loss.view(-1, 1).expand_as(x_tgt)
    probs = F.softmax(x_src, dim=1)
    pt = probs * x_tgt + (1 - probs) * (1 - x_tgt)
    loss = alpha * ((1 - pt) ** gamma) * loss * x_tgt
    # apply class_weight
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        avg_factor = (class_weight * x_tgt).sum()
        if not norm_loss_with_class_weight:
            avg_factor = None
    else:
        avg_factor = None
    # reduce loss with weight
    loss = reducelosswithweight(loss.sum(dim=1), weight, reduction, avg_factor)
    # return calculated loss
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


'''softmaxfocalloss'''
def softmaxfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean', norm_loss_with_class_weight=True):
    # mmcv.ops._softmax_focal_loss only accept x_tgt with class labels, e.g., (B, C, H, W) as inputs and (B, H, W) as targets
    if x_src.shape == x_tgt.shape:
        return pysoftmaxfocalloss(x_src, x_tgt, weight=weight, gamma=gamma, alpha=alpha, class_weight=class_weight, ignore_index=ignore_index, reduction=reduction)
    # convert x_src and x_tgt to tensors with shape (N, num_classes) and (N,)
    original_shape, num_classes = x_src.shape, x_src.size(1)
    assert num_classes > 1, 'num classes of inputs should be larger than 1'
    x_src = bchw2nc(x_src)
    x_tgt = x_tgt.view(-1).contiguous().long()
    ignore_index_mask = (x_tgt != ignore_index)
    x_src = x_src[ignore_index_mask]
    x_tgt = x_tgt[ignore_index_mask]
    # convert weight, mmcv.ops._softmax_focal_loss always return tensor with shape (N,)
    if weight is not None:
        weight = weight.view(-1).contiguous().float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    # calculate softmax focal loss
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        avg_factor = class_weight[x_tgt].sum()
        if not norm_loss_with_class_weight:
            avg_factor = None
    else:
        avg_factor = None
    loss = _softmax_focal_loss(x_src, x_tgt, gamma, alpha, class_weight, 'none')
    # reduce loss with weight
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    # return calculated loss
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


'''FocalLoss'''
class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, scale_factor=1.0, gamma=2, alpha=0.25, class_weight=None, reduction='mean', ignore_index=-100, lowest_loss_value=None, norm_loss_with_class_weight=True):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
        self.norm_loss_with_class_weight = norm_loss_with_class_weight
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert (x_src.shape == x_tgt.shape) or (x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:]), 'invalid shape of x_src or x_tgt'
        # calculate loss
        if self.use_sigmoid:
            if torch.cuda.is_available() and x_src.is_cuda and _sigmoid_focal_loss is not None:
                loss = sigmoidfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction, norm_loss_with_class_weight=self.norm_loss_with_class_weight)
            else:
                loss = pysigmoidfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction, norm_loss_with_class_weight=self.norm_loss_with_class_weight)
        else:
            if torch.cuda.is_available() and x_src.is_cuda and _softmax_focal_loss is not None:
                loss = softmaxfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction, norm_loss_with_class_weight=self.norm_loss_with_class_weight)
            else:
                loss = pysoftmaxfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction, norm_loss_with_class_weight=self.norm_loss_with_class_weight)
        # rescale loss
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        # return calculated loss
        return loss


'''unittest'''
if __name__ == '__main__':
    import numpy as np
    for use_sigmoid in [True, False]:
        for num_classes in [2, 21, 151]:
            # cuda or cpu
            print(f'*** TEST on CUDA and CPU (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
            # class weight
            print(f'*** TEST on CUDA and CPU with class weight (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            class_weight = np.random.rand(num_classes).tolist()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt_onehot))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt_onehot))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt_onehot.cuda()))
            # weight
            print(f'*** TEST on CUDA and CPU with weight (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            weight = torch.rand(batch_size, num_classes, h, w) if use_sigmoid else torch.rand(batch_size, h, w)
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt, weight))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot, weight))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous() if use_sigmoid else weight.view(-1).contiguous()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt, weight))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot, weight))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
            # ignore index
            print(f'*** TEST on CUDA and CPU with ignore index (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt_onehot))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt_onehot))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt.cuda()))
            print(FocalLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt_onehot.cuda()))