'''
Function:
    Build normalization
Author:
    Zhenchao Jin
'''
from torch.nn import Identity
from .groupnorm import GroupNorm
from .layernorm import LayerNorm
from .syncbatchnorm import SyncBatchNorm
from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from .instancenorm import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d


'''build normalization'''
def BuildNormalization(norm_type='batchnorm2d', instanced_params=(0, {}), only_get_all_supported=False, **kwargs):
    supported_dict = {
        'identity': Identity,
        'layernorm': LayerNorm,
        'groupnorm': GroupNorm,
        'batchnorm1d': BatchNorm1d,
        'batchnorm2d': BatchNorm2d,
        'batchnorm3d': BatchNorm3d,
        'syncbatchnorm': SyncBatchNorm,
        'instancenorm1d': InstanceNorm1d,
        'instancenorm2d': InstanceNorm2d,
        'instancenorm3d': InstanceNorm3d,
    }
    if only_get_all_supported: return list(supported_dict.values())
    assert norm_type in supported_dict, 'unsupport norm_type %s...' % norm_type
    norm_layer = supported_dict[norm_type](instanced_params[0], **instanced_params[1])
    return norm_layer