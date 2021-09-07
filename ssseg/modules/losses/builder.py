'''
Function:
    build the loss
Author:
    Zhenchao Jin
'''
from .diceloss import DiceLoss
from .lovaszloss import LovaszLoss
from .focalloss import SigmoidFocalLoss
from .celoss import CrossEntropyLoss, BinaryCrossEntropyLoss


'''build loss'''
def BuildLoss(loss_type, **kwargs):
    supported_losses = {
        'diceloss': DiceLoss,
        'lovaszloss': LovaszLoss,
        'celoss': CrossEntropyLoss,
        'sigmoidfocalloss': SigmoidFocalLoss,
        'binaryceloss': BinaryCrossEntropyLoss,
    }
    assert loss_type in supported_losses, 'unsupport loss type %s...' % loss_type
    return supported_losses[loss_type]