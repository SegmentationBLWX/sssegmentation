'''
Function:
    build for dataloader and model
Author:
    Zhenchao Jin
'''
from .distributed import *
from .nondistributed import *


'''build parallel dataloader'''
def BuildParallelDataloader(mode, dataset, cfg, **kwargs):
    supported_dict = {
        'distributed': DistributedDataloader,
        'nondistributed': NonDistributedDataloader
    }
    cfg = cfg[mode.lower()]
    assert cfg['type'] in supported_dict, 'unsupport dataloader type %s...' % cfg['type']
    return supported_dict[cfg['type']](dataset, cfg, **kwargs)


'''build parallel model'''
def BuildParallelModel(model, is_distributed=False, **kwargs):
    if is_distributed:
        return DistributedModel(model, **kwargs)
    else:
        return NonDistributedModel(model, **kwargs)