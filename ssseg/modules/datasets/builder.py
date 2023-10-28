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
from ..utils import BaseModuleBuilder
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
class DatasetBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'BaseDataset': BaseDataset, 'VOCDataset': VOCDataset, 'PascalContext59Dataset': PascalContext59Dataset, 'PascalContextDataset': PascalContextDataset,
        'COCODataset': COCODataset, 'COCOStuff10kDataset': COCOStuff10kDataset, 'COCOStuffDataset': COCOStuffDataset, 'CIHPDataset': CIHPDataset,
        'LIPDataset': LIPDataset, 'ATRDataset': ATRDataset, 'MHPv1Dataset': MHPv1Dataset, 'MHPv2Dataset': MHPv2Dataset, 'SuperviselyDataset': SuperviselyDataset,
        'HRFDataset': HRFDataset, 'ChaseDB1Dataset': ChaseDB1Dataset, 'STAREDataset': STAREDataset, 'DRIVEDataset': DRIVEDataset, 'SBUShadowDataset': SBUShadowDataset,
        'VSPWDataset': VSPWDataset, 'ADE20kDataset': ADE20kDataset, 'DarkZurichDataset': DarkZurichDataset, 'NighttimeDrivingDataset': NighttimeDrivingDataset,
        'CityScapesDataset': CityScapesDataset,
    }
    '''build'''
    def build(self, mode, logger_handle, dataset_cfg):
        dataset_cfg = copy.deepcopy(dataset_cfg)
        train_cfg, test_cfg = dataset_cfg.pop('train', {}), dataset_cfg.pop('test', {})
        dataset_cfg.update(train_cfg if mode == 'TRAIN' else test_cfg)
        module_cfg = {
            'mode': mode, 'logger_handle': logger_handle, 'dataset_cfg': dataset_cfg, 'type': dataset_cfg['type'],
        }
        return super().build(module_cfg)


'''BuildDataset'''
BuildDataset = DatasetBuilder().build