'''fcn_bisenetv2_cityscapes'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cityscapes',
    'rootdir': os.path.join(os.getcwd(), 'CityScapes'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (1024, 1024), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (1024, 1024), 'data_type': 'tensor'}),
]
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update({
    'type': 'sgd',
    'lr': 0.05,
    'momentum': 0.9,
    'weight_decay': 5e-4,
})
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 860,
    'min_lr': 1e-4,
    'warmup_cfg': {'type': 'linear', 'ratio': 0.1, 'iters': 1000},
})
# modify losses config
LOSSES_CFG = {
    'loss_aux1': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_aux2': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_aux3': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_aux4': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_cls': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
}
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 19,
    'backbone': {
        'type': None,
        'series': 'bisenetv2',
        'pretrained': False,
        'selected_indices': (0, 1, 2, 3, 4),
    },
    'head': {
        'in_channels': 128,
        'feats_channels': 1024,
        'dropout': 0.1,
        'num_convs': 1,
    },
    'auxiliary': [
        {'in_channels': 16, 'out_channels': 16, 'dropout': 0.1, 'num_convs': 2},
        {'in_channels': 32, 'out_channels': 64, 'dropout': 0.1, 'num_convs': 2},
        {'in_channels': 64, 'out_channels': 256, 'dropout': 0.1, 'num_convs': 2},
        {'in_channels': 128, 'out_channels': 1024, 'dropout': 0.1, 'num_convs': 2},
    ],
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'fcn_bisenetv2_cityscapes'
COMMON_CFG['logfilepath'] = 'fcn_bisenetv2_cityscapes/fcn_bisenetv2_cityscapes.log'
COMMON_CFG['resultsavepath'] = 'fcn_bisenetv2_cityscapes/fcn_bisenetv2_cityscapes_results.pkl'