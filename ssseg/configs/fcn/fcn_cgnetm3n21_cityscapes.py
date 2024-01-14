'''fcn_cgnetm3n21_cityscapes'''
import os
import copy
import torch
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
SEGMENTOR_CFG['dataset']['train']['data_pipelines'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [72.39239876, 82.90891754, 73.15835921], 'std': [1, 1, 1]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),
]
SEGMENTOR_CFG['dataset']['test']['data_pipelines'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [72.39239876, 82.90891754, 73.15835921], 'std': [1, 1, 1]}),
    ('ToTensor', {}),
]
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 340
SEGMENTOR_CFG['scheduler']['min_lr'] = 1e-4
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'Adam', 'lr': 0.001, 'eps': 1e-08, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'CGNet', 'structure_type': 'cgnetm3n21', 'pretrained': False, 'selected_indices': (1, 2),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 256, 'feats_channels': 256, 'dropout': 0, 'num_convs': 0,
}
SEGMENTOR_CFG['auxiliary'] = None
SEGMENTOR_CFG['losses'] = {
    'loss_cls': {
        'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean', 
        'weight': torch.tensor([2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352, 10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905, 10.347791, 6.3927646, 10.226669, 10.241062, 10.280587, 10.396974, 10.055647]),
    },
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")