'''define the config file for atr and resnet50os16'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'atr',
        'rootdir': 'data/ATR',
        'aug_opts': [('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': (0.75, 1.25)}),
                     ('RandomCrop', {'crop_size': (473, 473), 'one_category_max_ratio': 0.75}),
                     ('RandomFlip', {'flip_prob': 0.5, 'fix_ann_pairs': [(9, 10), (12, 13), (14, 15)]}),
                     ('RandomRotation', {'angle_upper': 30, 'rotation_prob': 0.6}),
                     ('PhotoMetricDistortion', {}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),
                     ('Padding', {'output_size': (473, 473), 'data_type': 'tensor'}),]
    }
)
DATASET_CFG['test'].update(
    {
        'type': 'atr',
        'rootdir': 'data/ATR',
        'aug_opts': [('Resize', {'output_size': (473, 473), 'keep_ratio': False, 'scale_range': None}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),]
    }
)
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
        'max_epochs': 150
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 18,
        'backbone': {
            'type': 'resnet50',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 16,
            'use_stem': True,
            'selected_indices': (0, 1, 2, 3),
        }
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'ce2p_resnet50os16_atr_train',
        'logfilepath': 'ce2p_resnet50os16_atr_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'ce2p_resnet50os16_atr_test',
        'logfilepath': 'ce2p_resnet50os16_atr_test/test.log',
        'resultsavepath': 'ce2p_resnet50os16_atr_test/ce2p_resnet50os16_atr_results.pkl'
    }
)