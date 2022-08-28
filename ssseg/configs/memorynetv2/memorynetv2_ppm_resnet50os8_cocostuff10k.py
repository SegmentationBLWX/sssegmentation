'''memorynetv2_ppm_resnet50os8_cocostuff10k'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cocostuff10k',
    'rootdir': os.path.join(os.getcwd(), 'COCOStuff10k'),
})
DATASET_CFG['test']['set'] = 'test'
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = {
    'type': 'sgd',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'params_rules': {},
}
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 110,
    'min_lr': 1e-4,
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 182,
    'backbone': {
        'type': 'resnet50',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
})
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cls': {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
}
SEGMENTOR_CFG['head']['context_within_image']['type'] = 'ppm'
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynetv2_ppm_resnet50os8_cocostuff10k'
COMMON_CFG['logfilepath'] = 'memorynetv2_ppm_resnet50os8_cocostuff10k/memorynetv2_ppm_resnet50os8_cocostuff10k.log'
COMMON_CFG['resultsavepath'] = 'memorynetv2_ppm_resnet50os8_cocostuff10k/memorynetv2_ppm_resnet50os8_cocostuff10k_results.pkl'