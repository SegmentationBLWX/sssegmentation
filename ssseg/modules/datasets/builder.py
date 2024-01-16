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
from .voc import VOCDataset, PascalContextDataset, PascalContext59Dataset
from .coco import COCOVOCSUBDataset, COCOStuffDataset, COCOStuff10kDataset


'''MultipleDataset'''
class MultipleDataset(BaseDataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        for key, value in dataset_cfg.copy().items():
            if not isinstance(value, dict) and 'type' not in value:
                dataset_cfg.pop(key)
        super(MultipleDataset, self).__init__(mode, logger_handle, list(dataset_cfg.values())[0])
        self.datasets = []
        self.num_classes = 0
        self.classnames = []
        self.clsid2clsnames = {}
        self.dataset_cfg = dataset_cfg
        # build all datasets
        for dataset_cfg_item in list(dataset_cfg.values()):
            self.datasets.append(BuildDataset(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg_item))
            seg_target_remapper = dataset_cfg_item['seg_target_remapper']
            for src_label, tgt_label in seg_target_remapper.items():
                self.clsid2clsnames[tgt_label] = self.datasets[-1].classnames[src_label]
                self.num_classes = max(self.num_classes, tgt_label)
        # auto fill some necessary attributes
        self.num_classes += 1
        for clsid in range(self.num_classes):
            self.classnames.append(self.clsid2clsnames[clsid])
        self.palette = BaseDataset.randompalette(self.num_classes)
        assert self.num_classes == len(self.classnames) and self.num_classes == len(self.palette)
    '''getitem'''
    def __getitem__(self, index):
        # obtain sample_meta
        pointer, sample_meta = 0, None
        for dataset_idx, dataset in enumerate(self.datasets):
            if index < len(dataset) + pointer:
                sample_meta = dataset[index - pointer]
                break
            else:
                pointer += len(dataset)
        assert sample_meta is not None
        # remap seg_target
        if 'seg_target' in sample_meta and sample_meta['seg_target'] is not None:
            seg_target = sample_meta['seg_target']
            seg_target_remapper = list(self.dataset_cfg.values())[dataset_idx]['seg_target_remapper']
            for src_label, tgt_label in seg_target_remapper.items():
                seg_target[seg_target == src_label] = tgt_label
            sample_meta['seg_target'] = seg_target
        # format data
        if 'id' in sample_meta:
            sample_meta['id'] = str(sample_meta['id'])
        # return
        return sample_meta
    '''len'''
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])


'''DatasetBuilder'''
class DatasetBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'BaseDataset': BaseDataset, 'VOCDataset': VOCDataset, 'PascalContext59Dataset': PascalContext59Dataset, 'PascalContextDataset': PascalContextDataset,
        'COCOVOCSUBDataset': COCOVOCSUBDataset, 'COCOStuff10kDataset': COCOStuff10kDataset, 'COCOStuffDataset': COCOStuffDataset, 'CIHPDataset': CIHPDataset,
        'LIPDataset': LIPDataset, 'ATRDataset': ATRDataset, 'MHPv1Dataset': MHPv1Dataset, 'MHPv2Dataset': MHPv2Dataset, 'SuperviselyDataset': SuperviselyDataset,
        'HRFDataset': HRFDataset, 'ChaseDB1Dataset': ChaseDB1Dataset, 'STAREDataset': STAREDataset, 'DRIVEDataset': DRIVEDataset, 'SBUShadowDataset': SBUShadowDataset,
        'VSPWDataset': VSPWDataset, 'ADE20kDataset': ADE20kDataset, 'DarkZurichDataset': DarkZurichDataset, 'NighttimeDrivingDataset': NighttimeDrivingDataset,
        'CityScapesDataset': CityScapesDataset, 'MultipleDataset': MultipleDataset,
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