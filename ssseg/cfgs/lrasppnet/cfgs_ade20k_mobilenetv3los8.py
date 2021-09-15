'''define the config file for ade20k and mobilenetv3los8'''
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
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'lrasppnet_mobilenetv3los8_ade20k_train',
        'logfilepath': 'lrasppnet_mobilenetv3los8_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'lrasppnet_mobilenetv3los8_ade20k_test',
        'logfilepath': 'lrasppnet_mobilenetv3los8_ade20k_test/test.log',
        'resultsavepath': 'lrasppnet_mobilenetv3los8_ade20k_test/lrasppnet_mobilenetv3los8_ade20k_results.pkl'
    }
)