'''define the config file for ade20k and mobilenetv3sos8'''
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
        'max_epochs': 390
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
            'type': 'mobilenetv3',
            'series': 'mobilenet',
            'pretrained': True,
            'outstride': 8,
            'arch_type': 'small',
            'out_indices': (0, 1, 12),
            'selected_indices': (0, 1, 2),
        },
        'aspp': {
            'in_channels_list': [16, 16, 576],
            'branch_channels_list': [32, 64],
            'out_channels': 128,
        },
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'lrasppnet_mobilenetv3sos8_ade20k_train',
        'logfilepath': 'lrasppnet_mobilenetv3sos8_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'lrasppnet_mobilenetv3sos8_ade20k_test',
        'logfilepath': 'lrasppnet_mobilenetv3sos8_ade20k_test/test.log',
        'resultsavepath': 'lrasppnet_mobilenetv3sos8_ade20k_test/lrasppnet_mobilenetv3sos8_ade20k_results.pkl'
    }
)