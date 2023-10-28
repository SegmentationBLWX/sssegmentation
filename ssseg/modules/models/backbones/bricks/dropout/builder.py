'''
Function:
    Implementation of DropoutBuilder and BuildDropout
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from .droppath import DropPath
from .....utils import BaseModuleBuilder


'''DropoutBuilder'''
class DropoutBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DropPath': DropPath, 'Dropout': nn.Dropout, 'Dropout2d': nn.Dropout2d, 'Dropout3d': nn.Dropout3d,
    }
    '''build'''
    def build(self, dropout_cfg):
        if dropout_cfg is None: return nn.Identity()
        return super().build(dropout_cfg)


'''BuildDropout'''
BuildDropout = DropoutBuilder().build