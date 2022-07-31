'''upernet_convnextbase_ade20k'''
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
    'max_epochs': 130,
    'min_lr': 0.0,
    'power': 1.0,
    'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'convnext_base',
        'series': 'convnext',
        'arch': 'base',
        'pretrained': True,
        'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0,
        'gap_before_final_norm': False,
        'selected_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm2d', 'eps': 1e-6},
    },
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'upernet_convnextbase_ade20k'
COMMON_CFG['logfilepath'] = 'upernet_convnextbase_ade20k/upernet_convnextbase_ade20k.log'
COMMON_CFG['resultsavepath'] = 'upernet_convnextbase_ade20k/upernet_convnextbase_ade20k_results.pkl'