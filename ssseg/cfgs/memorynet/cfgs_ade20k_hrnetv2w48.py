'''define the config file for ade20k and hrnetv2w48'''
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
        'type': 'sgd',
        'sgd': {
            'learning_rate': 0.004,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        },
        'max_epochs': 180
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
LOSSES_CFG.pop('loss_aux')
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 150,
        'backbone': {
            'type': 'hrnetv2_w48',
            'series': 'hrnet',
            'pretrained': True,
            'selected_indices': (0, 0, 0, 0),
        },
        'auxiliary': None,
    }
)
MODEL_CFG['memory']['in_channels'] = sum([48, 96, 192, 384])
MODEL_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.004
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'memorynet_hrnetv2w48_ade20k_train',
        'logfilepath': 'memorynet_hrnetv2w48_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'memorynet_hrnetv2w48_ade20k_test',
        'logfilepath': 'memorynet_hrnetv2w48_ade20k_test/test.log',
        'resultsavepath': 'memorynet_hrnetv2w48_ade20k_test/memorynet_hrnetv2w48_ade20k_results.pkl'
    }
)