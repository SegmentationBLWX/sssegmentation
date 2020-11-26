'''
Function:
    builder for parallel dataloader and parallel model
Author:
    Zhenchao Jin
'''
from .distributed import *
from .nondistributed import *


'''build parallel dataloader'''
def BuildParallelDataloader(mode, dataset, cfg, **kwargs):
    supported_dataloaders = {
        'distributed': DistributedDataloader,
        'nondistributed': NonDistributedDataloader
    }
    cfg = cfg[mode.lower()]
    assert cfg['type'] in supported_dataloaders, 'unsupport dataloader type %s...' % cfg['type']
    return supported_dataloaders[cfg['type']](dataset, cfg, **kwargs)


'''build parallel model'''
def BuildParallelModel(model, is_distributed=False, **kwargs):
    if is_distributed:
        return DistributedModel(model, **kwargs)
    else:
        return NonDistributedModel(model, **kwargs)