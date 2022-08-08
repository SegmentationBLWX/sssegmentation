'''pspnet_mobilenetv2os8_voc'''
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
        'type': 'mobilenetv2',
        'series': 'mobilenet',
        'pretrained': True,
        'outstride': 8,
        'selected_indices': (2, 3),
    },
    'head': {
        'in_channels': 320,
        'feats_channels': 512,
        'pool_scales': [1, 2, 3, 6],
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 96,
        'out_channels': 512,
        'dropout': 0.1,
    },
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'pspnet_mobilenetv2os8_voc'
COMMON_CFG['logfilepath'] = 'pspnet_mobilenetv2os8_voc/pspnet_mobilenetv2os8_voc.log'
COMMON_CFG['resultsavepath'] = 'pspnet_mobilenetv2os8_voc/pspnet_mobilenetv2os8_voc_results.pkl'