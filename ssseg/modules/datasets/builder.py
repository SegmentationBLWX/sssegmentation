'''
Function:
    Implementation of BuildDataset
Author:
    Zhenchao Jin
'''
import copy
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
from .darkzurich import DarkZurichDataset
from .supervisely import SuperviselyDataset
from .mhp import MHPv1Dataset, MHPv2Dataset
from .nighttimedriving import NighttimeDrivingDataset
from .coco import COCODataset, COCOStuffDataset, COCOStuff10kDataset
from .voc import VOCDataset, PascalContextDataset, PascalContext59Dataset


'''BuildDataset'''
def BuildDataset(mode, logger_handle, dataset_cfg):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    # supported datasets
    supported_datasets = {
        'VOCDataset': VOCDataset, 'LIPDataset': LIPDataset, 'ATRDataset': ATRDataset, 'HRFDataset': HRFDataset,
        'BaseDataset': BaseDataset, 'COCODataset': COCODataset, 'VSPWDataset': VSPWDataset, 'CIHPDataset': CIHPDataset,
        'MHPv1Dataset': MHPv1Dataset, 'MHPv2Dataset': MHPv2Dataset, 'STAREDataset': STAREDataset, 'DRIVEDataset': DRIVEDataset,
        'ADE20kDataset': ADE20kDataset, 'ChaseDB1Dataset': ChaseDB1Dataset, 'COCOStuffDataset': COCOStuffDataset, 
        'SBUShadowDataset': SBUShadowDataset, 'CityScapesDataset': CityScapesDataset, 'DarkZurichDataset': DarkZurichDataset, 
        'SuperviselyDataset': SuperviselyDataset, 'COCOStuff10kDataset': COCOStuff10kDataset, 'PascalContextDataset': PascalContextDataset, 
        'PascalContext59Dataset': PascalContext59Dataset, 'NighttimeDrivingDataset': NighttimeDrivingDataset,
    }
    # build dataset
    train_cfg = dataset_cfg.pop('train')
    test_cfg = dataset_cfg.pop('test')
    if mode == 'TRAIN':
        dataset_cfg.update(train_cfg)
    else:
        dataset_cfg.update(test_cfg)
    dataset = supported_datasets[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
    # return
    return dataset