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
from .....utils import BaseModuleBuilder


'''NormalizationBuilder'''
class NormalizationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 
        'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 
        'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d, 'GRN': GRN,
    }
    for norm_type in ['LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d', 'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d']:
        if hasattr(nn, norm_type):
            REGISTERED_MODULES[norm_type] = getattr(nn, norm_type)
    '''build'''
    def build(self, placeholder, norm_cfg):
        if norm_cfg is None: return nn.Identity()
        norm_cfg = copy.deepcopy(norm_cfg)
        norm_type = norm_cfg.pop('type')
        if norm_type in ['GroupNorm']:
            normalization = self.REGISTERED_MODULES[norm_type](num_channels=placeholder, **norm_cfg)
        else:
            normalization = self.REGISTERED_MODULES[norm_type](placeholder, **norm_cfg)
        return normalization
    '''isnorm'''
    @staticmethod
    def isnorm(module, norm_list=None):
        if norm_list is None:
            norm_list = (
                nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.SyncBatchNorm,
            )
        return isinstance(module, norm_list)


'''BuildNormalization'''
BuildNormalization = NormalizationBuilder().build