'''segformer_mitb4_ade20k'''
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
        'type': 'mit-b4',
        'series': 'mit',
        'pretrained': True,
        'pretrained_model_path': 'mit_b4.pth',
        'selected_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'head': {
        'in_channels_list': [64, 128, 320, 512],
        'feats_channels': 256,
        'dropout': 0.1,
    },
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'segformer_mitb4_ade20k'
COMMON_CFG['logfilepath'] = 'segformer_mitb4_ade20k/segformer_mitb4_ade20k.log'
COMMON_CFG['resultsavepath'] = 'segformer_mitb4_ade20k/segformer_mitb4_ade20k_results.pkl'