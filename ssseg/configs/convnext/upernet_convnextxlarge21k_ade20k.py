'''upernet_convnextxlarge21k_ade20k'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'ade20k',
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'),
})
DATASET_CFG['train'].update({
    'aug_opts': [
        ('Resize', {'output_size': (2560, 640), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
        ('RandomCrop', {'crop_size': (640, 640), 'one_category_max_ratio': 0.75}),
        ('RandomFlip', {'flip_prob': 0.5}),
        ('PhotoMetricDistortion', {}),
        ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
        ('ToTensor', {}),
        ('Padding', {'output_size': (640, 640), 'data_type': 'tensor'}),
    ],
})
DATASET_CFG['test'].update({
    'aug_opts': [
        ('Resize', {'output_size': (2560, 640), 'keep_ratio': True, 'scale_range': None}),
        ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
        ('ToTensor', {}),
    ],
})
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 130,
    'min_lr': 0.0,
    'power': 1.0,
    'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'convnext_xlarge_21k',
        'series': 'convnext',
        'arch': 'xlarge',
        'pretrained': True,
        'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0,
        'gap_before_final_norm': False,
        'selected_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm2d', 'eps': 1e-6},
    },
    'head': {
        'in_channels_list': [256, 512, 1024, 2048],
        'feats_channels': 1024,
        'pool_scales': [1, 2, 3, 6],
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024,
        'out_channels': 512,
        'dropout': 0.1,
    }
})
# modify inference config
INFERENCE_CFG = {
    'mode': 'slide',
    'opts': {'cropsize': (640, 640), 'stride': (426, 426)}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': True
    }
}
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'upernet_convnextxlarge21k_ade20k'
COMMON_CFG['logfilepath'] = 'upernet_convnextxlarge21k_ade20k/upernet_convnextxlarge21k_ade20k.log'
COMMON_CFG['resultsavepath'] = 'upernet_convnextxlarge21k_ade20k/upernet_convnextxlarge21k_ade20k_results.pkl'