'''
Function:
    Build the loss
Author:
    Zhenchao Jin
'''
from .l1loss import L1Loss
from .klloss import KLDivLoss
from .diceloss import DiceLoss
from .lovaszloss import LovaszLoss
from .focalloss import SigmoidFocalLoss
from .cosinesimilarityloss import CosineSimilarityLoss
from .celoss import CrossEntropyLoss, BinaryCrossEntropyLoss


'''BuildLoss'''
def BuildLoss(loss_type):
    supported_losses = {
        'l1loss': L1Loss,
        'diceloss': DiceLoss,
        'kldivloss': KLDivLoss,
        'lovaszloss': LovaszLoss,
        'celoss': CrossEntropyLoss,
        'sigmoidfocalloss': SigmoidFocalLoss,
        'binaryceloss': BinaryCrossEntropyLoss,
        'cosinesimilarityloss': CosineSimilarityLoss,
    }
    return supported_losses[loss_type]