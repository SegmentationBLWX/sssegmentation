'''
Function:
    Implementation of CrossEntropyLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import reducelosswithweight, bchw2nc


'''crossentropy'''
def crossentropy(x_src, x_tgt, weight=None, class_weight=None, reduction='mean', ignore_index=-100, label_smoothing=None, norm_loss_with_class_weight=True):
    # convert x_src and x_tgt, mainly for adapting to low-version Pytorch
    original_shape, num_classes = x_src.shape, x_src.size(1)
    assert num_classes > 1, 'num classes of inputs should be larger than 1'
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index = -100
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
        weight = weight.view(-1).contiguous()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    # calculate cross entropy loss
    if label_smoothing is None:
        loss = F.cross_entropy(x_src, x_tgt.argmax(dim=1), reduction='none')
    else:
        loss = F.cross_entropy(x_src, x_tgt.argmax(dim=1), reduction='none', label_smoothing=label_smoothing)
    loss = loss.view(-1, 1).expand_as(x_tgt)
    loss = loss * x_tgt
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


'''binarycrossentropy'''
def binarycrossentropy(x_src, x_tgt, weight=None, class_weight=None, reduction='mean', ignore_index=-100, norm_loss_with_class_weight=True):
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
    # calculate binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(x_src, x_tgt, reduction='none')
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


'''CrossEntropyLoss'''
class CrossEntropyLoss(nn.Module):
    def __init__(self, use_sigmoid=False, reduction='mean', class_weight=None, scale_factor=1.0, lowest_loss_value=None, label_smoothing=None, ignore_index=-100, norm_loss_with_class_weight=True):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.class_weight = class_weight
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.norm_loss_with_class_weight = norm_loss_with_class_weight
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert (x_src.shape == x_tgt.shape) or (x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:]), 'invalid shape of x_src or x_tgt'
        # calculate loss
        if self.use_sigmoid:
            loss = binarycrossentropy(x_src, x_tgt, weight=weight, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index, norm_loss_with_class_weight=self.norm_loss_with_class_weight)
        else:
            loss = crossentropy(x_src, x_tgt, weight=weight, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing, norm_loss_with_class_weight=self.norm_loss_with_class_weight)
        # rescale loss
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        # return calculated loss
        return loss


'''unittest'''
if __name__ == '__main__':
    import numpy as np
    # use_sigmoid = False
    use_sigmoid = False
    for num_classes in [2, 21, 151]:
        # cuda or cpu
        print(f'*** TEST on CUDA and CPU (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        print('Expected loss: ', F.cross_entropy(x_src, x_tgt, reduction='mean'))
        # class weight
        print(f'*** TEST on CUDA and CPU with class weight (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        class_weight = np.random.rand(num_classes).tolist()
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        print('Expected loss: ', F.cross_entropy(x_src, x_tgt, reduction='mean', weight=torch.tensor(class_weight)))
        # weight
        print(f'*** TEST on CUDA and CPU with weight (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        weight = torch.rand(batch_size, num_classes, h, w) if use_sigmoid else torch.rand(batch_size, h, w)
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous() if use_sigmoid else weight.view(-1).contiguous()
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
        print('Expected loss: ', (F.cross_entropy(x_src, x_tgt, reduction='none') * weight).mean())
        # ignore index
        print(f'*** TEST on CUDA and CPU with ignore index (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print('Expected loss: ', F.cross_entropy(x_src, x_tgt, reduction='mean', ignore_index=2))
    # use_sigmoid = True
    use_sigmoid = True
    for num_classes in [2, 21, 151]:
        # cuda or cpu
        print(f'*** TEST on CUDA and CPU (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        print('Expected loss: ', F.binary_cross_entropy_with_logits(x_src, x_tgt_onehot.float(), reduction='mean'))
        # class weight
        print(f'*** TEST on CUDA and CPU with class weight (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        class_weight = np.random.rand(num_classes).tolist()
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print(CrossEntropyLoss(reduction='mean', class_weight=class_weight, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda()))
        avg_factor = (torch.tensor(class_weight).reshape(1, -1) * x_tgt_onehot).sum()
        print('Expected loss: ', F.binary_cross_entropy_with_logits(x_src, x_tgt_onehot.float(), reduction='none', weight=torch.tensor(class_weight).reshape(1, -1)).sum() / avg_factor)
        # weight
        print(f'*** TEST on CUDA and CPU with weight (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        weight = torch.rand(batch_size, num_classes, h, w) if use_sigmoid else torch.rand(batch_size, h, w)
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous() if use_sigmoid else weight.view(-1).contiguous()
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src, x_tgt_onehot, weight))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print(CrossEntropyLoss(reduction='mean', use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
        print('Expected loss: ', (F.binary_cross_entropy_with_logits(x_src, x_tgt_onehot.float(), reduction='none') * weight).mean())
        # ignore index
        print(f'*** TEST on CUDA and CPU with ignore index (use_sigmoid={use_sigmoid}) ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src, x_tgt))
        print(CrossEntropyLoss(reduction='mean', ignore_index=2, use_sigmoid=use_sigmoid)(x_src.cuda(), x_tgt.cuda()))
        print('Expected loss: ', F.binary_cross_entropy_with_logits(x_src[:, torch.arange(num_classes) != 2], x_tgt_onehot[:, torch.arange(num_classes) != 2].float(), reduction='mean'))