'''
Function:
    Implementation of DatasetBuilder and BuildDataset
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


'''DatasetBuilder'''
class DatasetBuilder():
    REGISTERED_DATASETS = {
        'VOCDataset': VOCDataset, 'LIPDataset': LIPDataset, 'ATRDataset': ATRDataset, 'HRFDataset': HRFDataset,
        'BaseDataset': BaseDataset, 'COCODataset': COCODataset, 'VSPWDataset': VSPWDataset, 'CIHPDataset': CIHPDataset,
        'MHPv1Dataset': MHPv1Dataset, 'MHPv2Dataset': MHPv2Dataset, 'STAREDataset': STAREDataset, 'DRIVEDataset': DRIVEDataset,
        'ADE20kDataset': ADE20kDataset, 'ChaseDB1Dataset': ChaseDB1Dataset, 'COCOStuffDataset': COCOStuffDataset, 
        'SBUShadowDataset': SBUShadowDataset, 'CityScapesDataset': CityScapesDataset, 'DarkZurichDataset': DarkZurichDataset, 
        'SuperviselyDataset': SuperviselyDataset, 'COCOStuff10kDataset': COCOStuff10kDataset, 'PascalContextDataset': PascalContextDataset, 
        'PascalContext59Dataset': PascalContext59Dataset, 'NighttimeDrivingDataset': NighttimeDrivingDataset,
    }
    def __init__(self, require_register_datasets=None, require_update_datasets=None):
        if require_register_datasets and isinstance(require_register_datasets, dict):
            for dataset_type, dataset in require_register_datasets.items():
                self.register(dataset_type, dataset)
        if require_update_datasets and isinstance(require_update_datasets, dict):
            for dataset_type, dataset in require_update_datasets.items():
                self.update(dataset_type, dataset)
    '''build'''
    def build(self, mode, logger_handle, dataset_cfg):
        dataset_cfg = copy.deepcopy(dataset_cfg)
        train_cfg = dataset_cfg.pop('train')
        test_cfg = dataset_cfg.pop('test')
        if mode == 'TRAIN':
            dataset_cfg.update(train_cfg)
        else:
            dataset_cfg.update(test_cfg)
        dataset = self.REGISTERED_DATASETS[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        return dataset
    '''register'''
    def register(self, dataset_type, dataset):
        assert dataset_type not in self.REGISTERED_DATASETS
        self.REGISTERED_DATASETS[dataset_type] = dataset
    '''update'''
    def update(self, dataset_type, dataset):
        assert dataset_type in self.REGISTERED_DATASETS
        self.REGISTERED_DATASETS[dataset_type] = dataset


'''BuildDataset'''
BuildDataset = DatasetBuilder().build