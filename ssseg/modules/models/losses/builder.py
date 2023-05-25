'''
Function:
    Implementation of BuildLoss
Author:
    Zhenchao Jin
'''
import copy
from .l1loss import L1Loss
from .klloss import KLDivLoss
from .diceloss import DiceLoss
from .lovaszloss import LovaszLoss
from .focalloss import SigmoidFocalLoss
from .cosinesimilarityloss import CosineSimilarityLoss
from .celoss import CrossEntropyLoss, BinaryCrossEntropyLoss


'''BuildLoss'''
def BuildLoss(loss_cfg):
    loss_cfg = copy.deepcopy(loss_cfg)
    # supported losses
    supported_losses = {
        'L1Loss': L1Loss, 'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss,
        'CrossEntropyLoss': CrossEntropyLoss, 'SigmoidFocalLoss': SigmoidFocalLoss,
        'CosineSimilarityLoss': CosineSimilarityLoss, 'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
    }
    # build loss
    loss_type = loss_cfg.pop('type')
    loss_func = supported_losses[loss_type](**loss_cfg)
    # return
    return loss_func