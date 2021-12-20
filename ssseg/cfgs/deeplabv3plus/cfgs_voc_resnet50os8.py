'''define the config file for voc and resnet50os8'''
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
OPTIMIZER_CFG.update(
    {
        'max_epochs': 60,
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update(
    {
        'num_classes': 21,
        'backbone': {
            'type': 'resnet50',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 8,
            'use_stem': True,
            'selected_indices': (0, 1, 2, 3),
        },
        'aspp': {
            'in_channels': 2048,
            'out_channels': 512,
            'dilations': [1, 12, 24, 36],
        },
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'deeplabv3plus_resnet50os8_voc_train',
        'logfilepath': 'deeplabv3plus_resnet50os8_voc_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'deeplabv3plus_resnet50os8_voc_test',
        'logfilepath': 'deeplabv3plus_resnet50os8_voc_test/test.log',
        'resultsavepath': 'deeplabv3plus_resnet50os8_voc_test/deeplabv3plus_resnet50os8_voc_results.pkl'
    }
)