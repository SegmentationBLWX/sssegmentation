'''
Function:
    build normalization layer
Author:
    Zhenchao Jin
'''
from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from mmcv.utils.parrots_wrapper import SyncBatchNorm as DistSyncBatchNorm
from .syncbatchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d


'''build normalization layer'''
def BuildNormalizationLayer(norm_type='syncbatchnorm2d', instanced_params=(0, {}), only_get_all_supported=False):
    supported_dict = {
        'batchnorm1d': BatchNorm1d,
        'batchnorm2d': BatchNorm2d,
        'batchnorm3d': BatchNorm3d,
        'syncbatchnorm1d': SynchronizedBatchNorm1d,
        'syncbatchnorm2d': SynchronizedBatchNorm2d,
        'syncbatchnorm3d': SynchronizedBatchNorm3d,
        'distsyncbatchnorm': DistSyncBatchNorm
    }
    if only_get_all_supported: return list(supported_dict.values())
    assert norm_type in supported_dict, 'unsupport norm_type %s...' % norm_type
    norm_layer = supported_dict[norm_type](instanced_params[0], **instanced_params[1])
    if norm_type in ['distsyncbatchnorm']: norm_layer._specify_ddp_gpu_num(1)
    return norm_layer