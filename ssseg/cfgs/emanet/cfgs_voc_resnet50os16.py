'''define the config file for voc and resnet50os16'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'voc',
        'set': 'trainaug',
        'rootdir': 'data/VOCdevkit/VOC2012',
    }
)
DATASET_CFG['test'].update(
    {
        'type': 'voc',
        'rootdir': 'data/VOCdevkit/VOC2012',
    }
)
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
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 21,
        'backbone': {
            'type': 'resnet50',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 16,
            'use_stem': True
        },
    }
)
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'emanet_resnet50os16_voc_train',
        'logfilepath': 'emanet_resnet50os16_voc_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'emanet_resnet50os16_voc_test',
        'logfilepath': 'emanet_resnet50os16_voc_test/test.log',
        'resultsavepath': 'emanet_resnet50os16_voc_test/emanet_resnet50os16_voc_results.pkl'
    }
)