'''
Function:
    builder for building instanced dataset class for various datasets.
Author:
    Zhenchao Jin
'''
from .voc import *
from .lip import *
from .coco import *
from .ade20k import *
from .sbushadow import *
from .cityscapes import *
from .supervisely import *


'''build dataset'''
def BuildDataset(mode, logger_handle, dataset_cfg, **kwargs):
    dataset_cfg = dataset_cfg[mode.lower()]
    supported_datasets = {
        'voc': VOCDataset,
        'lip': LIPDataset,
        'coco': COCODataset,
        'ade20k': ADE20kDataset,
        'cocostuff': COCOStuffDataset,
        'sbushadow': SBUShadowDataset,
        'voccontext': VOCContextDataset,
        'cityscapes': CityScapesDataset,
        'supervisely': SuperviselyDataset,
    }
    assert dataset_cfg['type'] in supported_datasets, 'unsupport dataset type %s...' % dataset_cfg['type']
    if kwargs.get('get_basedataset', False): return BaseDataset(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg, **kwargs)
    dataset = supported_datasets[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg, **kwargs)
    return dataset