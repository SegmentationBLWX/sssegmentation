'''
Function:
    Implementation of FocalLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import reducelosswithweight
try:
    from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
except:
    _sigmoid_focal_loss = None


'''pysigmoidfocalloss'''
def pysigmoidfocalloss(x_src, x_tgt, x_tgt_onehot=None, weight=None, gamma=2.0, alpha=0.5, class_weight=None, valid_mask=None, reduction='mean'):
    # calculate focal loss
    if isinstance(alpha, list):
        alpha = x_src.new_tensor(alpha).reshape(1, -1)
    x_src_sigmoid, x_tgt = x_src.sigmoid(), x_tgt.type_as(x_src)
    one_minus_pt = (1 - x_src_sigmoid) * x_tgt + x_src_sigmoid * (1 - x_tgt)
    focal_weight = (alpha * x_tgt + (1 - alpha) * (1 - x_tgt)) * one_minus_pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(x_src, x_tgt, reduction='none') * focal_weight
    # reweighted loss
    final_weight = torch.ones_like(x_src)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * x_src.new_tensor(class_weight).reshape(1, -1)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    # reduce loss with weight
    loss = reducelosswithweight(loss, final_weight, reduction, valid_mask.sum())
    # return calculated loss
    return loss


'''sigmoidfocalloss'''
def sigmoidfocalloss(x_src, x_tgt, x_tgt_onehot=None, weight=None, gamma=2.0, alpha=0.5, class_weight=None, valid_mask=None, reduction='mean'):
    # calculate focal loss
    final_weight = torch.ones_like(x_src)
    if isinstance(alpha, list):
        loss = _sigmoid_focal_loss(x_src.contiguous(), x_tgt.contiguous(), gamma, 0.5, None, 'none') * 2
        alpha = x_src.new_tensor(alpha).reshape(1, -1)
        final_weight = final_weight * (alpha * x_tgt_onehot + (1 - alpha) * (1 - x_tgt_onehot))
    else:
        loss = _sigmoid_focal_loss(x_src.contiguous(), x_tgt.contiguous(), gamma, alpha, None, 'none')
    # reweighted loss
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * x_src.new_tensor(class_weight).reshape(1, -1)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    # reduce loss with weight
    loss = reducelosswithweight(loss, final_weight, reduction, valid_mask.sum())
    # return calculated loss
    return loss


'''FocalLoss'''
class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, scale_factor=1.0, gamma=2, alpha=0.25, class_weight=None, reduction='mean', ignore_index=255, lowest_loss_value=None):
        super(FocalLoss, self).__init__()
        assert use_sigmoid, 'only support SigmoidFocalLoss'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt, weight=None):
        # assert
        assert (x_src.shape == x_tgt.shape) or (x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:]), 'invalid shape of x_src or x_tgt'
        if weight is None: weight = torch.ones_like(x_src)
        assert (x_src.shape == weight.shape) or (x_src.size(0) == weight.size(0) and x_src.shape[2:] == weight.shape[1:]), 'invalid shape of weight'
        # convert x_src and x_tgt: (N, num_classes) or (N,)
        original_shape = x_src.shape
        if x_src.shape == x_tgt.shape:
            x_tgt = x_tgt.transpose(0, 1)
            x_tgt = x_tgt.reshape(x_tgt.size(0), -1)
            x_tgt = x_tgt.transpose(0, 1).contiguous()
        else:
            x_tgt = x_tgt.view(-1).contiguous()
            valid_mask = (x_tgt != self.ignore_index).view(-1, 1)
            x_tgt = torch.where(x_tgt == self.ignore_index, x_tgt.new_tensor(0), x_tgt)
        x_src = x_src.transpose(0, 1)
        x_src = x_src.reshape(x_src.size(0), -1)
        x_src = x_src.transpose(0, 1).contiguous()
        # convert weight: (N, num_classes) or (N,)
        if original_shape == weight.shape:
            weight = weight.transpose(0, 1)
            weight = weight.reshape(weight.size(0), -1)
            weight = weight.transpose(0, 1).contiguous()
        else:
            weight = weight.view(-1).contiguous()
        # calculate loss
        if self.use_sigmoid:
            num_classes = x_src.size(1)
            if torch.cuda.is_available() and x_src.is_cuda and _sigmoid_focal_loss is not None:
                if x_tgt.dim() == 1:
                    x_tgt_onehot = F.one_hot(x_tgt, num_classes=num_classes+1)
                    if num_classes == 1:
                        x_tgt_onehot = x_tgt_onehot[:, 1]
                        x_tgt = 1 - x_tgt
                    else:
                        x_tgt_onehot = x_tgt_onehot[:, :num_classes]
                else:
                    x_tgt_onehot = x_tgt
                    x_tgt = x_tgt.argmax(dim=1)
                    valid_mask = (x_tgt != self.ignore_index).view(-1, 1)
                loss_func = sigmoidfocalloss
            else:
                x_tgt_onehot = None
                if x_tgt.dim() == 1:
                    x_tgt = F.one_hot(x_tgt, num_classes=num_classes+1)
                    if num_classes == 1:
                        x_tgt = x_tgt[:, 1].unsqueeze(-1)
                    else:
                        x_tgt = x_tgt[:, :num_classes]
                else:
                    valid_mask = (x_tgt.argmax(dim=1) != self.ignore_index).view(-1, 1)
                loss_func = pysigmoidfocalloss
            loss = loss_func(
                x_src=x_src, x_tgt=x_tgt, x_tgt_onehot=x_tgt_onehot, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, valid_mask=valid_mask, reduction=self.reduction
            )
            if self.reduction == 'none':
                loss = loss.transpose(0, 1)
                loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
                loss = loss.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError('only support SigmoidFocalLoss')
        # rescale loss
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        # return calculated loss
        return loss


'''unittest'''
if __name__ == '__main__':
    import numpy as np
    for num_classes in [2, 21, 151]:
        # cuda or cpu
        print('*** TEST on CUDA and CPU ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        class_weight = np.random.rand(num_classes).tolist()
        alpha_list = np.random.rand(num_classes).tolist()
        weight = torch.rand(batch_size, num_classes, h, w)
        print(FocalLoss(reduction='mean')(x_src, x_tgt))
        print(FocalLoss(reduction='mean')(x_src, x_tgt_onehot))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda()))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(FocalLoss(reduction='mean')(x_src, x_tgt))
        print(FocalLoss(reduction='mean')(x_src, x_tgt_onehot))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda()))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda()))
        # class weight
        print('*** TEST on CUDA and CPU with class weight ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        class_weight = np.random.rand(num_classes).tolist()
        alpha_list = np.random.rand(num_classes).tolist()
        weight = torch.rand(batch_size, num_classes, h, w)
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src, x_tgt))
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src, x_tgt_onehot))
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src.cuda(), x_tgt.cuda()))
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src, x_tgt))
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src, x_tgt_onehot))
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src.cuda(), x_tgt.cuda()))
        print(FocalLoss(reduction='mean', class_weight=class_weight)(x_src.cuda(), x_tgt_onehot.cuda()))
        # alpha list
        print('*** TEST on CUDA and CPU with alpha list ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        class_weight = np.random.rand(num_classes).tolist()
        alpha_list = np.random.rand(num_classes).tolist()
        weight = torch.rand(batch_size, num_classes, h, w)
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src, x_tgt))
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src, x_tgt_onehot))
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src.cuda(), x_tgt.cuda()))
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src.cuda(), x_tgt_onehot.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src, x_tgt))
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src, x_tgt_onehot))
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src.cuda(), x_tgt.cuda()))
        print(FocalLoss(reduction='mean', alpha=alpha_list)(x_src.cuda(), x_tgt_onehot.cuda()))
        # weight
        print('*** TEST on CUDA and CPU with weight ***')
        batch_size, h, w = 4, 128, 128
        x_src, x_tgt = torch.rand(batch_size, num_classes, h, w), torch.randint(0, num_classes, (batch_size, h, w))
        x_tgt_onehot = F.one_hot(x_tgt, num_classes).permute(0, 3, 1, 2).contiguous()
        class_weight = np.random.rand(num_classes).tolist()
        alpha_list = np.random.rand(num_classes).tolist()
        weight = torch.rand(batch_size, num_classes, h, w)
        print(FocalLoss(reduction='mean')(x_src, x_tgt, weight))
        print(FocalLoss(reduction='mean')(x_src, x_tgt_onehot, weight))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))
        x_src = x_src.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        x_tgt = x_tgt.reshape(-1)
        x_tgt_onehot = x_tgt_onehot.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, num_classes).contiguous()
        print(FocalLoss(reduction='mean')(x_src, x_tgt, weight))
        print(FocalLoss(reduction='mean')(x_src, x_tgt_onehot, weight))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt.cuda(), weight.cuda()))
        print(FocalLoss(reduction='mean')(x_src.cuda(), x_tgt_onehot.cuda(), weight.cuda()))