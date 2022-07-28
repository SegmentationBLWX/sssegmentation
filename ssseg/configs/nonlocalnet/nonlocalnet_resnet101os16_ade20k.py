'''nonlocalnet_resnet101os16_ade20k'''
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
        'outstride': 16,
        'use_stem': True,
        'selected_indices': (2, 3),
    },
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'nonlocalnet_resnet101os16_ade20k'
COMMON_CFG['logfilepath'] = 'nonlocalnet_resnet101os16_ade20k/nonlocalnet_resnet101os16_ade20k.log'
COMMON_CFG['resultsavepath'] = 'nonlocalnet_resnet101os16_ade20k/nonlocalnet_resnet101os16_ade20k_results.pkl'