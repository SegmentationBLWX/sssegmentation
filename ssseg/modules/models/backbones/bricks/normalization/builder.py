'''
Function:
    Implementation of NormalizationBuilder and BuildNormalization
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from .grn import GRN
from .layernorm2d import LayerNorm2d


'''NormalizationBuilder'''
class NormalizationBuilder():
    REGISTERED_NORMALIZATIONS = {
        'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 
        'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 
        'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d, 'GRN': GRN,
    }
    for norm_type in ['LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d', 'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d']:
        if hasattr(nn, norm_type):
            REGISTERED_NORMALIZATIONS[norm_type] = getattr(nn, norm_type)
    def __init__(self, require_register_normalizations=None, require_update_normalizations=None):
        if require_register_normalizations and isinstance(require_register_normalizations, dict):
            for norm_type, norm_module in require_register_normalizations.items():
                self.register(norm_type, norm_module)
        if require_update_normalizations and isinstance(require_update_normalizations, dict):
            for norm_type, norm_module in require_update_normalizations.items():
                self.update(norm_type, norm_module)
    '''build'''
    def build(self, placeholder, norm_cfg):
        if norm_cfg is None: return nn.Identity()
        norm_cfg = copy.deepcopy(norm_cfg)
        norm_type = norm_cfg.pop('type')
        if norm_type in ['GroupNorm']:
            normalization = self.REGISTERED_NORMALIZATIONS[norm_type](num_channels=placeholder, **norm_cfg)
        else:
            normalization = self.REGISTERED_NORMALIZATIONS[norm_type](placeholder, **norm_cfg)
        return normalization
    '''register'''
    def register(self, norm_type, norm_module):
        assert norm_type not in self.REGISTERED_NORMALIZATIONS
        self.REGISTERED_NORMALIZATIONS[norm_type] = norm_module
    '''update'''
    def update(self, norm_type, norm_module):
        assert norm_type in self.REGISTERED_NORMALIZATIONS
        self.REGISTERED_NORMALIZATIONS[norm_type] = norm_module
    '''isnorm'''
    @staticmethod
    def isnorm(module):
        norm_list = (
            nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.SyncBatchNorm,
        )
        return isinstance(module, norm_list)


'''BuildNormalization'''
BuildNormalization = NormalizationBuilder().build