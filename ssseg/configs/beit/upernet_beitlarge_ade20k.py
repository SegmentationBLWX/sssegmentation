'''upernet_beitlarge_ade20k'''
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
    'lr': 3e-5,
    'betas': (0.9, 0.999),
    'weight_decay': 0.05,
    'params_rules': {'type': 'layerdecay', 'num_layers': 24, 'decay_rate': 0.95, 'decay_type': 'layer_wise_vit'},
}
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 130,
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'beit_large_patch16_224_pt22k_ft22k',
        'series': 'beit',
        'pretrained': True,
        'selected_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'head': {
        'feature2pyramid': {
            'embed_dim': 1024, 
            'rescales': [4, 2, 1, 0.5],
        },
        'in_channels_list': [1024, 1024, 1024, 1024],
        'feats_channels': 1024,
        'pool_scales': [1, 2, 3, 6],
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024,
        'out_channels': 512,
        'dropout': 0.1,
    }
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'upernet_beitlarge_ade20k'
COMMON_CFG['logfilepath'] = 'upernet_beitlarge_ade20k/upernet_beitlarge_ade20k.log'
COMMON_CFG['resultsavepath'] = 'upernet_beitlarge_ade20k/upernet_beitlarge_ade20k_results.pkl'