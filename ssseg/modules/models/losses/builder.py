'''
Function:
    Implementation of LossBuilder and BuildLoss
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


'''LossBuilder'''
class LossBuilder():
    REGISTERED_LOSSES = {
        'L1Loss': L1Loss, 'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss,
        'CrossEntropyLoss': CrossEntropyLoss, 'SigmoidFocalLoss': SigmoidFocalLoss,
        'CosineSimilarityLoss': CosineSimilarityLoss, 'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
    }
    def __init__(self, require_register_losses=None, require_update_losses=None):
        if require_register_losses and isinstance(require_register_losses, dict):
            for loss_type, loss_func in require_register_losses.items():
                self.register(loss_type, loss_func)
        if require_update_losses and isinstance(require_update_losses, dict):
            for loss_type, loss_func in require_update_losses.items():
                self.update(loss_type, loss_func)
    '''build'''
    def build(self, loss_cfg):
        loss_cfg = copy.deepcopy(loss_cfg)
        loss_type = loss_cfg.pop('type')
        loss_func = self.REGISTERED_LOSSES[loss_type](**loss_cfg)
        return loss_func
    '''register'''
    def register(self, loss_type, loss_func):
        assert loss_type not in self.REGISTERED_LOSSES
        self.REGISTERED_LOSSES[loss_type] = loss_func
    '''update'''
    def update(self, loss_type, loss_func):
        assert loss_type in self.REGISTERED_LOSSES
        self.REGISTERED_LOSSES[loss_type] = loss_func


'''BuildLoss'''
BuildLoss = LossBuilder().build