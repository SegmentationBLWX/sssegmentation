'''upernet_swinbase_ade20k'''
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
OPTIMIZER_CFG = {
    'type': 'adamw',
    'lr': 0.00006,
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,
    'params_rules': {'backbone_net_zerowd': (1.0, 0.0), 'others': (1.0, 1.0)},
}
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
        'type': 'swin_base_patch4_window12_384_22k',
        'series': 'swin',
        'pretrained': True,
        'selected_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm'},
    },
    'head': {
        'in_channels_list': [128, 256, 512, 1024],
        'feats_channels': 512,
        'pool_scales': [1, 2, 3, 6],
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 512,
        'out_channels': 512,
        'dropout': 0.1,
    }
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'upernet_swinbase_ade20k'
COMMON_CFG['logfilepath'] = 'upernet_swinbase_ade20k/upernet_swinbase_ade20k.log'
COMMON_CFG['resultsavepath'] = 'upernet_swinbase_ade20k/upernet_swinbase_ade20k_results.pkl'