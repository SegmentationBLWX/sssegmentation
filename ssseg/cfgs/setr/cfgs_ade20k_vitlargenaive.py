'''define the config file for ade20k and ViT-Large'''
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
            'type': 'jx_vit_large_p16_384',
            'series': 'vit',
            'img_size': (512, 512),
            'drop_rate': 0.,
            'out_indices': (9, 14, 19, 23),
            'norm_cfg': {'type': 'layernorm', 'opts': {'eps': 1e-6}},
            'pretrained': True,
            'selected_indices': (0, 1, 2, 3),
        },
        'decoder': {
            'in_channels': 1024,
            'out_channels': 256,
            'dropout': 0,
            'num_convs': 2,
            'scale_factor': 4,
            'kernel_size': 3,
        },
        'auxiliary': [
            {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
            {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
            {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        ],
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'setrnaive_vitlarge_ade20k_train',
        'logfilepath': 'setrnaive_vitlarge_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'setrnaive_vitlarge_ade20k_test',
        'logfilepath': 'setrnaive_vitlarge_ade20k_test/test.log',
        'resultsavepath': 'setrnaive_vitlarge_ade20k_test/setrnaive_vitlarge_ade20k_results.pkl'
    }
)