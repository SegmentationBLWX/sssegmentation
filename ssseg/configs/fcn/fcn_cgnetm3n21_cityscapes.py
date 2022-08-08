'''fcn_cgnetm3n21_cityscapes'''
import os
import torch
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cityscapes',
    'rootdir': os.path.join(os.getcwd(), 'CityScapes'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [72.39239876, 82.90891754, 73.15835921], 'std': [1, 1, 1]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),
]
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [72.39239876, 82.90891754, 73.15835921], 'std': [1, 1, 1]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = {
    'type': 'adam',
    'lr': 0.001,
    'eps': 1e-08,
    'weight_decay': 5e-4,
    'params_rules': {},
}
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 340,
    'min_lr': 1e-4,
})
# modify losses config
LOSSES_CFG = {
    'loss_cls': {
        'celoss': {
            'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean', 
            'weight': torch.tensor([
                2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 
                9.369352, 10.289121, 9.953208, 4.3097677, 9.490387, 
                7.674431, 9.396905, 10.347791, 6.3927646, 10.226669, 
                10.241062, 10.280587, 10.396974, 10.055647,
            ])
        }
    },
}
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 19,
    'backbone': {
        'type': None,
        'series': 'cgnet',
        'pretrained': False,
        'selected_indices': (1, 2),
    },
    'head': {
        'in_channels': 256,
        'feats_channels': 256,
        'dropout': 0,
        'num_convs': 0,
    },
    'auxiliary': None,
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'fcn_cgnetm3n21_cityscapes'
COMMON_CFG['logfilepath'] = 'fcn_cgnetm3n21_cityscapes/fcn_cgnetm3n21_cityscapes.log'
COMMON_CFG['resultsavepath'] = 'fcn_cgnetm3n21_cityscapes/fcn_cgnetm3n21_cityscapes_results.pkl'