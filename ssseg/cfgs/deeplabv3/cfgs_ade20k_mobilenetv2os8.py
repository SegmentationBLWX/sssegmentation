'''define the config file for ade20k and mobilenetv2os8'''
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
        'num_classes': 150,
        'backbone': {
            'type': 'mobilenetv2',
            'series': 'mobilenet',
            'pretrained': True,
            'outstride': 8,
            'selected_indices': (2, 3),
        },
        'aspp': {
            'in_channels': 320,
            'out_channels': 512,
            'dilations': [1, 12, 24, 36],
        },
        'auxiliary': {
            'in_channels': 96,
            'out_channels': 512,
            'dropout': 0.1,
        },
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'deeplabv3_mobilenetv2os8_ade20k_train',
        'logfilepath': 'deeplabv3_mobilenetv2os8_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'deeplabv3_mobilenetv2os8_ade20k_test',
        'logfilepath': 'deeplabv3_mobilenetv2os8_ade20k_test/test.log',
        'resultsavepath': 'deeplabv3_mobilenetv2os8_ade20k_test/deeplabv3_mobilenetv2os8_ade20k_results.pkl'
    }
)