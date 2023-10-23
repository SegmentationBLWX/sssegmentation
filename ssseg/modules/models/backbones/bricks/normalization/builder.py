'''
Function:
    Implementation of BuildNormalization
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from .layernorm2d import LayerNorm2d


'''BuildNormalization'''
def BuildNormalization(placeholder, norm_cfg):
    if norm_cfg is None: return nn.Identity()
    norm_cfg = copy.deepcopy(norm_cfg)
    # supported normalizations
    supported_normalizations = {
        'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 
        'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 
        'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d,
    }
    for norm_type in ['LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d', 'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d']:
        if hasattr(nn, norm_type):
            supported_normalizations[norm_type] = getattr(nn, norm_type)
    # build normalization
    norm_type = norm_cfg.pop('type')
    if norm_type in ['GroupNorm']:
        normalization = supported_normalizations[norm_type](num_channels=placeholder, **norm_cfg)
    else:
        normalization = supported_normalizations[norm_type](placeholder, **norm_cfg)
    # return
    return normalization