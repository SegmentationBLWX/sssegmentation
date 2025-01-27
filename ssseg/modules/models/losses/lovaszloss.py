'''
Function:
    Implementation of LovaszLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import reducelosswithweight, bchw2nc


'''lovaszgrad'''
def lovaszgrad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


'''_sigmoidlovaszloss'''
def _sigmoidlovaszloss(x_src, x_tgt, classes='present', class_weight=None, ignore_index=-100):
    # convert x_src and x_tgt to tensor with shape (N, num_classes)
    num_classes = x_src.shape[1]
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_tgt = bchw2nc(x_tgt).float()
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
    # select x_src, x_tgt and class_weight according to classes
    if classes == 'present':
        classes = []
        for cls_id in list(range(x_src.shape[1])):
            if cls_id == ignore_index and ignore_index_mask is None:
                continue
            fg = x_tgt[:, cls_id]
            if fg.sum() == 0:
                continue
            classes.append(cls_id)
    elif classes == 'all':
        classes = torch.arange(x_src.shape[1])
    classes = list(set(classes))
    classes = [x for x in classes if x != ignore_index]
    x_src, x_tgt = x_src[:, classes], x_tgt[:, classes]
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        class_weight = class_weight[classes]
        class_weight = class_weight.reshape(1, -1)
    # calculate sigmoid lovasz loss
    signs = 2. * x_tgt.float() - 1.
    errors = (1. - x_src * signs)
    if class_weight is not None:
        errors = errors * class_weight
    errors, x_tgt = errors.view(-1), x_tgt.view(-1)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = x_tgt[perm.data]
    grad = lovaszgrad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    # return calculated loss
    return loss


'''sigmoidlovaszloss'''
def sigmoidlovaszloss(x_src, x_tgt, weight=None, classes='present', per_image=False, class_weight=None, reduction='mean', ignore_index=-100):
    # calculate sigmoid lovasz loss
    x_src = x_src.sigmoid()
    if per_image:
        loss = [_sigmoidlovaszloss(x_src[idx].unsqueeze(0), x_tgt[idx].unsqueeze(0), classes=classes, class_weight=class_weight, ignore_index=ignore_index) for idx in range(x_src.shape[0])]
        loss = torch.stack(loss).reshape(-1)
    else:
        loss = _sigmoidlovaszloss(x_src, x_tgt, classes=classes, class_weight=class_weight, ignore_index=ignore_index)
    # reduce loss with weight
    loss = reducelosswithweight(loss, weight, reduction, None)
    # return calculated loss
    return loss


'''_softmaxlovaszloss'''
def _softmaxlovaszloss(x_src, x_tgt, classes='present', class_weight=None, ignore_index=-100):
    # convert x_src and x_tgt to tensor with shape (N, num_classes)
    num_classes = x_src.shape[1]
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_tgt = bchw2nc(x_tgt).float()
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
    # calculate softmax lovasz loss
    loss, loss_weight = [], []
    classes_to_sum = list(range(num_classes)) if classes in ['all', 'present'] else list(set(classes))
    for cls_id in classes_to_sum:
        if cls_id == ignore_index and ignore_index_mask is None:
            continue
        fg = x_tgt[:, cls_id]
        if (classes == 'present' and fg.sum() == 0):
            continue
        class_prob = x_src[:, cls_id]
        errors = (fg - class_prob).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        loss.append(torch.dot(errors_sorted, lovaszgrad(fg_sorted)))
        if class_weight is not None:
            loss_weight.append(class_weight[cls_id])
    loss = torch.stack(loss).reshape(-1)
    # apply class weight
    if class_weight is not None:
        loss_weight = loss.new_tensor(loss_weight).reshape(-1)
        loss = (loss * loss_weight).sum() / loss_weight.sum()
    else:
        loss = loss.mean()
    # return calculated loss
    return loss


'''softmaxlovaszloss'''
def softmaxlovaszloss(x_src, x_tgt, weight=None, classes='present', per_image=False, class_weight=None, reduction='mean', ignore_index=-100):
    # calculate softmax lovasz loss
    x_src = F.softmax(x_src, dim=1) if x_src.shape[1] > 1 else x_src
    if per_image:
        loss = [_softmaxlovaszloss(x_src[idx].unsqueeze(0), x_tgt[idx].unsqueeze(0), classes=classes, class_weight=class_weight, ignore_index=ignore_index) for idx in range(x_src.shape[0])]
        loss = torch.stack(loss).reshape(-1)
    else:
        loss = _softmaxlovaszloss(x_src, x_tgt, classes=classes, class_weight=class_weight, ignore_index=ignore_index)
    # reduce loss with weight
    loss = reducelosswithweight(loss, weight, reduction, None)
    # return calculated loss
    return loss


'''LovaszLoss'''
class LovaszLoss(nn.Module):
    def __init__(self, use_sigmoid=False, classes='present', per_image=False, reduction='mean', class_weight=None, scale_factor=1.0, ignore_index=-100, lowest_loss_value=None):
        super(LovaszLoss, self).__init__()
        assert classes in ('all', 'present') or (isinstance(classes, (list, tuple)) and all(isinstance(elem, int) for elem in classes))
        self.use_sigmoid = use_sigmoid
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.class_weight = class_weight
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert (x_src.shape == x_tgt.shape) or (x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:]), 'invalid shape of x_src or x_tgt'
        # calculate loss
        if self.use_sigmoid:
            loss = sigmoidlovaszloss(x_src, x_tgt, weight=weight, classes=self.classes, per_image=self.per_image, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index)
        else:
            loss = softmaxlovaszloss(x_src, x_tgt, weight=weight, classes=self.classes, per_image=self.per_image, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index)
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
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
            # class weight
            print(f'*** TEST on CUDA and CPU with class weight (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            class_weight = np.random.rand(num_classes).tolist()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, class_weight=class_weight)(x_src.cuda(), x_tgt_onehot.cuda()))
            # weight
            print(f'*** TEST on CUDA and CPU with weight (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            weight = torch.rand(batch_size)
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src, x_tgt, weight))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src, x_tgt_onehot, weight))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
            # ignore index
            print(f'*** TEST on CUDA and CPU with ignore index (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, ignore_index=2)(x_src.cuda(), x_tgt_onehot.cuda()))
            # classes
            print(f'*** TEST on CUDA and CPU with classes (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src.cuda(), x_tgt_onehot.cuda()))
            x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            x_tgt = x_tgt.reshape(-1)
            x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, classes='all')(x_src.cuda(), x_tgt_onehot.cuda()))
            # per_image
            print(f'*** TEST on CUDA and CPU with per_image (use_sigmoid={use_sigmoid}) ***')
            batch_size, h, w = 4, 128, 128
            x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
            x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src, x_tgt))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src, x_tgt_onehot))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src.cuda(), x_tgt.cuda()))
            print(LovaszLoss(reduction='mean', use_sigmoid=use_sigmoid, per_image=True)(x_src.cuda(), x_tgt_onehot.cuda()))