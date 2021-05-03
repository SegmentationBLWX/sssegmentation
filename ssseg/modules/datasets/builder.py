'''
Function:
    builder for building instanced dataset class for various datasets.
Author:
    Zhenchao Jin
'''
from .lip import LIPDataset
from .atr import ATRDataset
from .base import BaseDataset
from .cihp import CIHPDataset
from .ade20k import ADE20kDataset
from .sbushadow import SBUShadowDataset
from .cityscapes import CityScapesDataset
from .supervisely import SuperviselyDataset
from .mhp import MHPv1Dataset, MHPv2Dataset
from .coco import COCODataset, COCOStuffDataset, COCOStuff10kDataset
from .voc import VOCDataset, PascalContextDataset, PascalContext59Dataset


'''build dataset'''
def BuildDataset(mode, logger_handle, dataset_cfg, **kwargs):
    dataset_cfg = dataset_cfg[mode.lower()]
    supported_datasets = {
        'voc': VOCDataset,
        'atr': ATRDataset,
        'lip': LIPDataset,
        'coco': COCODataset,
        'cihp': CIHPDataset,
        'mhpv1': MHPv1Dataset,
        'mhpv2': MHPv2Dataset,
        'ade20k': ADE20kDataset,
        'cocostuff': COCOStuffDataset,
        'sbushadow': SBUShadowDataset,
        'cityscapes': CityScapesDataset,
        'supervisely': SuperviselyDataset,
        'cocostuff10k': COCOStuff10kDataset,
        'pascalcontext': PascalContextDataset,
        'pascalcontext59': PascalContext59Dataset,
    }
    assert dataset_cfg['type'] in supported_datasets, 'unsupport dataset type %s...' % dataset_cfg['type']
    if kwargs.get('get_basedataset', False): return BaseDataset(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg, **kwargs)
    dataset = supported_datasets[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg, **kwargs)
    return dataset