'''setrnaive_vitlarge_ade20k'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'ade20k',
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'),
})
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 130
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'jx_vit_large_p16_384',
        'series': 'vit',
        'img_size': (512, 512),
        'drop_rate': 0.,
        'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
        'pretrained': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels_list': [1024, 1024, 1024, 1024],
        'feats_channels': 256,
        'dropout': 0,
        'num_convs': 2,
        'scale_factor': 4,
        'kernel_size': 3,
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'auxiliary': [
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
    ],
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'setrnaive_vitlarge_ade20k'
COMMON_CFG['logfilepath'] = 'setrnaive_vitlarge_ade20k/setrnaive_vitlarge_ade20k.log'
COMMON_CFG['resultsavepath'] = 'setrnaive_vitlarge_ade20k/setrnaive_vitlarge_ade20k_results.pkl'