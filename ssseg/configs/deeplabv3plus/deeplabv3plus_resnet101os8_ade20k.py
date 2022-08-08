'''deeplabv3plus_resnet101os8_ade20k'''
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
        'type': 'resnet101',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels': [256, 2048],
        'feats_channels': 512,
        'shortcut_channels': 48,
        'dilations': [1, 12, 24, 36],
        'dropout': 0.1,
    },
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'deeplabv3plus_resnet101os8_ade20k'
COMMON_CFG['logfilepath'] = 'deeplabv3plus_resnet101os8_ade20k/deeplabv3plus_resnet101os8_ade20k.log'
COMMON_CFG['resultsavepath'] = 'deeplabv3plus_resnet101os8_ade20k/deeplabv3plus_resnet101os8_ade20k_results.pkl'