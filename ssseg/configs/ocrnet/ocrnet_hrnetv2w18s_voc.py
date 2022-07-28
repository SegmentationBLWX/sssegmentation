'''ocrnet_hrnetv2w18s_voc'''
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
        'type': 'hrnetv2_w18_small',
        'series': 'hrnet',
        'pretrained': True,
        'selected_indices': (0, 0),
    },
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'ocrnet_hrnetv2w18s_voc'
COMMON_CFG['logfilepath'] = 'ocrnet_hrnetv2w18s_voc/ocrnet_hrnetv2w18s_voc.log'
COMMON_CFG['resultsavepath'] = 'ocrnet_hrnetv2w18s_voc/ocrnet_hrnetv2w18s_voc_results.pkl'