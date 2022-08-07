'''
Function:
    Build the dataset
Author:
    Zhenchao Jin
'''
from .lip import LIPDataset
from .atr import ATRDataset
from .hrf import HRFDataset
from .base import BaseDataset
from .vspw import VSPWDataset
from .cihp import CIHPDataset
from .stare import STAREDataset
from .drive import DRIVEDataset
from .ade20k import ADE20kDataset
from .chasedb1 import ChaseDB1Dataset
from .sbushadow import SBUShadowDataset
from .cityscapes import CityScapesDataset
from .supervisely import SuperviselyDataset
from .mhp import MHPv1Dataset, MHPv2Dataset
from .coco import COCODataset, COCOStuffDataset, COCOStuff10kDataset
from .voc import VOCDataset, PascalContextDataset, PascalContext59Dataset


'''BuildDataset'''
def BuildDataset(mode, logger_handle, dataset_cfg):
    # supported datasets
    supported_datasets = {
        'voc': VOCDataset,
        'lip': LIPDataset,
        'atr': ATRDataset,
        'hrf': HRFDataset,
        'base': BaseDataset,
        'coco': COCODataset,
        'vspw': VSPWDataset,
        'cihp': CIHPDataset,
        'mhpv1': MHPv1Dataset,
        'mhpv2': MHPv2Dataset,
        'stare': STAREDataset,
        'drive': DRIVEDataset,
        'ade20k': ADE20kDataset,
        'chasedb1': ChaseDB1Dataset,
        'cocostuff': COCOStuffDataset,
        'sbushadow': SBUShadowDataset,
        'cityscapes': CityScapesDataset,
        'supervisely': SuperviselyDataset,
        'cocostuff10k': COCOStuff10kDataset,
        'pascalcontext': PascalContextDataset,
        'pascalcontext59': PascalContext59Dataset,
    }
    # parse
    cfg = dataset_cfg[mode.lower()].copy()
    if 'train' in dataset_cfg: dataset_cfg.pop('train')
    if 'test' in dataset_cfg: dataset_cfg.pop('test')
    dataset_cfg.update(cfg)
    assert dataset_cfg['type'] in supported_datasets, 'unsupport dataset type %s' % dataset_cfg['type']
    # return
    dataset = supported_datasets[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
    return dataset