'''define the config file for cocostuff10k and hrnetv2w48'''
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
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'type': 'sgd',
        'sgd': {
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4,
        },
        'max_epochs': 110
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
LOSSES_CFG.pop('loss_aux')
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 182,
        'backbone': {
            'type': 'hrnetv2_w48',
            'series': 'hrnet',
            'pretrained': True,
            'selected_indices': (0, 0, 0, 0),
        },
        'auxiliary': None,
    }
)
MODEL_CFG['memory']['use_loss'] = False
MODEL_CFG['memory']['in_channels'] = sum([48, 96, 192, 384])
MODEL_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.001 * 0.9
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'memorynet_hrnetv2w48_cocostuff10k_train',
        'logfilepath': 'memorynet_hrnetv2w48_cocostuff10k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'memorynet_hrnetv2w48_cocostuff10k_test',
        'logfilepath': 'memorynet_hrnetv2w48_cocostuff10k_test/test.log',
        'resultsavepath': 'memorynet_hrnetv2w48_cocostuff10k_test/memorynet_hrnetv2w48_cocostuff10k_results.pkl'
    }
)