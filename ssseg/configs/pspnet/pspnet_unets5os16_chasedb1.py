'''pspnet_unets5os16_chasedb1'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'chasedb1',
    'rootdir': os.path.join(os.getcwd(), 'CHASE_DB1'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (960, 999), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (128, 128), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (128, 128), 'data_type': 'tensor'}),
]
DATASET_CFG['train']['repeat_times'] = 35000
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (960, 999), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 1,
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 2,
    'backbone': {
        'type': None,
        'series': 'unet',
        'pretrained': False,
        'selected_indices': (3, 4),
    },
    'head': {
        'in_channels': 64,
        'feats_channels': 16,
        'pool_scales': [1, 2, 3, 6],
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 128,
        'out_channels': 64,
        'dropout': 0.1,
    },
})
# modify inference config
INFERENCE_CFG = {
    'mode': 'slide',
    'opts': {'cropsize': (128, 128), 'stride': (85, 85)}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': True
    },
    'metric_list': ['dice', 'mdice'],
}
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'pspnet_unets5os16_chasedb1'
COMMON_CFG['logfilepath'] = 'pspnet_unets5os16_chasedb1/pspnet_unets5os16_chasedb1.log'
COMMON_CFG['resultsavepath'] = 'pspnet_unets5os16_chasedb1/pspnet_unets5os16_chasedb1_results.pkl'