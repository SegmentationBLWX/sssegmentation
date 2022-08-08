'''deeplabv3plus_resnest101os8_voc'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'voc',
    'rootdir': os.path.join(os.getcwd(), 'VOCdevkit/VOC2012'),
})
DATASET_CFG['train']['set'] = 'trainaug'
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 60,
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 21,
    'backbone': {
        'type': 'resnest101',
        'series': 'resnest',
        'pretrained': True,
        'outstride': 8,
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
COMMON_CFG['work_dir'] = 'deeplabv3plus_resnest101os8_voc'
COMMON_CFG['logfilepath'] = 'deeplabv3plus_resnest101os8_voc/deeplabv3plus_resnest101os8_voc.log'
COMMON_CFG['resultsavepath'] = 'deeplabv3plus_resnest101os8_voc/deeplabv3plus_resnest101os8_voc_results.pkl'