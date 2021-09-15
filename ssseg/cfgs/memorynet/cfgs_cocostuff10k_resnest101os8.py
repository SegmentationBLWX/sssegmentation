'''define the config file for cocostuff10k and resnest101os8'''
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
DATALOADER_CFG['train'].update(
    {
        'batch_size': 32,
    }
)
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'type': 'sgd',
        'sgd': {
            'learning_rate': 0.001,
            'min_lr': 1e-4,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        },
        'max_epochs': 150
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 182,
        'backbone': {
            'type': 'resnest101',
            'series': 'resnest',
            'pretrained': True,
            'outstride': 8,
            'selected_indices': (0, 1, 2, 3),
        },
    }
)
MODEL_CFG['memory']['use_loss'] = False
MODEL_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.001 * 0.9
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'memorynet_resnest101os8_cocostuff10k_train',
        'logfilepath': 'memorynet_resnest101os8_cocostuff10k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'memorynet_resnest101os8_cocostuff10k_test',
        'logfilepath': 'memorynet_resnest101os8_cocostuff10k_test/test.log',
        'resultsavepath': 'memorynet_resnest101os8_cocostuff10k_test/memorynet_resnest101os8_cocostuff10k_results.pkl'
    }
)