'''
Function:
    Implementation of SigmoidFocalLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
try:
    from mmcv.ops import sigmoid_focal_loss
except:
    sigmoid_focal_loss = None


'''SigmoidFocalLoss'''
class SigmoidFocalLoss(nn.Module):
    def __init__(self, scale_factor=1.0, gamma=2, alpha=0.25, weight=None, reduction='mean', ignore_index=None, lowest_loss_value=None):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, x_src, x_tgt):
        # assert
        assert (x_src.dim() == 4 and x_tgt.dim() == 3) or (x_src.dim() == 2 and x_tgt.dim() == 1)
        # convert x_src and x_tgt as sigmoid_focal_loss in mmcv only support (len(x_src.shape) == 2 and len(x_tgt.shape == 1))
        if len(x_src.shape) == 4 and len(x_tgt.shape == 3):
            num_classes = x_src.size(1)
            x_src = x_src.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
            x_tgt = x_tgt.reshape(-1)
        # fetch attributes
        alpha, gamma, weight, lowest_loss_value = self.alpha, self.gamma, self.weight, self.lowest_loss_value
        scale_factor, reduction, ignore_index = self.scale_factor, self.reduction, self.ignore_index
        # filter according to ignore_index
        if ignore_index is not None:
            num_classes = x_src.size(1)
            valid_mask = (x_tgt != ignore_index)
            x_src, x_tgt = x_src[valid_mask].view(-1, num_classes), x_tgt[valid_mask].view(-1)
        # calculate loss, sigmoid_focal_loss requires all input x_tgt as torch.LongTensor
        loss = sigmoid_focal_loss(x_src, x_tgt.long(), gamma, alpha, weight, reduction)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss