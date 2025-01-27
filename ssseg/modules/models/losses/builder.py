'''
Function:
    Implementation of LossBuilder and BuildLoss
Author:
    Zhenchao Jin
'''
from .l1loss import L1Loss
from .mseloss import MSELoss
from .diceloss import DiceLoss
from .kldivloss import KLDivLoss
from .focalloss import FocalLoss
from .lovaszloss import LovaszLoss
from ...utils import BaseModuleBuilder
from .crossentropyloss import CrossEntropyLoss
from .cosinesimilarityloss import CosineSimilarityLoss


'''LossBuilder'''
class LossBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'L1Loss': L1Loss, 'MSELoss': MSELoss, 'FocalLoss': FocalLoss, 'CosineSimilarityLoss': CosineSimilarityLoss, 
        'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss, 'CrossEntropyLoss': CrossEntropyLoss, 
    }
    '''build'''
    def build(self, loss_cfg):
        return super().build(loss_cfg)


'''BuildLoss'''
BuildLoss = LossBuilder().build