'''
Function:
    Build the loss
Author:
    Zhenchao Jin
'''
from .klloss import KLDivLoss
from .diceloss import DiceLoss
from .lovaszloss import LovaszLoss
from .focalloss import SigmoidFocalLoss
from .celoss import CrossEntropyLoss, BinaryCrossEntropyLoss


'''BuildLoss'''
def BuildLoss(loss_type):
    supported_losses = {
        'diceloss': DiceLoss,
        'kldivloss': KLDivLoss,
        'lovaszloss': LovaszLoss,
        'celoss': CrossEntropyLoss,
        'sigmoidfocalloss': SigmoidFocalLoss,
        'binaryceloss': BinaryCrossEntropyLoss,
    }
    return supported_losses[loss_type]