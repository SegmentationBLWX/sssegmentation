'''
Function:
    Implementation of BuildDropout
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from .droppath import DropPath


'''DropoutBuilder'''
class DropoutBuilder():
    REGISTERED_DROPOUTS = {
        'DropPath': DropPath, 'Dropout': nn.Dropout, 'Dropout2d': nn.Dropout2d, 'Dropout3d': nn.Dropout3d,
    }
    def __init__(self, require_register_dropouts=None, require_update_dropouts=None):
        if require_register_dropouts and isinstance(require_register_dropouts, dict):
            for dropout_type, dropout_module in require_register_dropouts.items():
                self.register(dropout_type, dropout_module)
        if require_update_dropouts and isinstance(require_update_dropouts, dict):
            for dropout_type, dropout_module in require_update_dropouts.items():
                self.update(dropout_type, dropout_module)
    '''build'''
    def build(self, dropout_cfg):
        if dropout_cfg is None: return nn.Identity()
        dropout_cfg = copy.deepcopy(dropout_cfg)
        dropout_type = dropout_cfg.pop('type')
        dropout = self.REGISTERED_DROPOUTS[dropout_type](**dropout_cfg)
        return dropout
    '''register'''
    def register(self, dropout_type, dropout_module):
        assert dropout_type not in self.REGISTERED_DROPOUTS
        self.REGISTERED_DROPOUTS[dropout_type] = dropout_module
    '''update'''
    def update(self, dropout_type, dropout_module):
        assert dropout_type in self.REGISTERED_DROPOUTS
        self.REGISTERED_DROPOUTS[dropout_type] = dropout_module


'''BuildDropout'''
BuildDropout = DropoutBuilder().build