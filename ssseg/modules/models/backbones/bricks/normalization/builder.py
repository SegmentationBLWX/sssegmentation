'''
Function:
    Build normalization
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn


'''BuildNormalization'''
def BuildNormalization(norm_cfg, only_get_all_supported=False):
    supported_normalizations = {
        'identity': [nn.Identity, None],
        'layernorm': [nn.LayerNorm, 'normalized_shape'],
        'groupnorm': [nn.GroupNorm, 'num_channels'],
        'batchnorm1d': [nn.BatchNorm1d, 'num_features'],
        'batchnorm2d': [nn.BatchNorm2d, 'num_features'],
        'batchnorm3d': [nn.BatchNorm3d, 'num_features'],
        'syncbatchnorm': [nn.SyncBatchNorm, 'num_features'],
        'instancenorm1d': [nn.InstanceNorm1d, 'num_features'],
        'instancenorm2d': [nn.InstanceNorm2d, 'num_features'],
        'instancenorm3d': [nn.InstanceNorm3d, 'num_features'],
    }
    if only_get_all_supported: 
        return list(supported_normalizations.values())
    selected_norm_func = supported_normalizations[norm_cfg['type']]
    norm_cfg = copy.deepcopy(norm_cfg)
    norm_cfg.pop('type')
    placeholder = norm_cfg.pop('placeholder')
    if selected_norm_func[-1] is not None:
        norm_cfg[selected_norm_func[1]] = placeholder
    return selected_norm_func[0](**norm_cfg)