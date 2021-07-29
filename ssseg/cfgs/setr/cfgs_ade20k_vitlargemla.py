'''define the config file for ade20k and ViT-Large'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'ade20k',
        'rootdir': 'data/ADE20k',
    }
)
DATASET_CFG['test'].update(
    {
        'type': 'ade20k',
        'rootdir': 'data/ADE20k',
    }
)
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
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'type': 'setrmla',
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
        'normlayer': {
            'in_channels_list': [1024, 1024, 1024, 1024],
            'type': 'layernorm', 
            'opts': {'eps': 1e-6},
        },
        'mla': {
            'in_channels_list': (1024, 1024, 1024, 1024),
            'out_channels': 256,
        },
        'decoder': {
            'in_channels_list': (256, 256, 256, 256),
            'mla_channels': 128,
            'out_channels': 512,
            'scale_factor': 4,
            'dropout': 0,
        },
        'auxiliary': [
            {'in_channels': 256, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
            {'in_channels': 256, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
            {'in_channels': 256, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        ],
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'setrmla_vitlarge_ade20k_train',
        'logfilepath': 'setrmla_vitlarge_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'setrmla_vitlarge_ade20k_test',
        'logfilepath': 'setrmla_vitlarge_ade20k_test/test.log',
        'resultsavepath': 'setrmla_vitlarge_ade20k_test/setrmla_vitlarge_ade20k_results.pkl'
    }
)