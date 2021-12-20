'''define the config file for ade20k and resnet101os16'''
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
OPTIMIZER_CFG.update(
    {
        'max_epochs': 130
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update(
    {
        'num_classes': 150,
        'backbone': {
            'type': 'resnet101',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 16,
            'use_stem': True,
            'selected_indices': (2, 3),
        },
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'emanet_resnet101os16_ade20k_train',
        'logfilepath': 'emanet_resnet101os16_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'emanet_resnet101os16_ade20k_test',
        'logfilepath': 'emanet_resnet101os16_ade20k_test/test.log',
        'resultsavepath': 'emanet_resnet101os16_ade20k_test/emanet_resnet101os16_ade20k_results.pkl'
    }
)