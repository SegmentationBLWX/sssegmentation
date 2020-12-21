'''define the config file for ade20k and resnet50os8'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'ade20k',
        'rootdir': '/data/ADE20k',
    }
)
DATASET_CFG['test'].update(
    {
        'type': 'ade20k',
        'rootdir': '/data/ADE20k',
    }
)
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'max_epochs': 130
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 150,
        'backbone': {
                'type': 'resnet50',
                'series': 'resnet',
                'pretrained': True,
                'outstride': 8,
                'is_use_stem': True
            },
    }
)
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'ccnet_resnet50os8_ade20k_train',
        'logfilepath': 'ccnet_resnet50os8_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'ccnet_resnet50os8_ade20k_test',
        'logfilepath': 'ccnet_resnet50os8_ade20k_test/test.log',
        'resultsavepath': 'ccnet_resnet50os8_ade20k_test/ccnet_resnet50os8_ade20k_results.pkl'
    }
)