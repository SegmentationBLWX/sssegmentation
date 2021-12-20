'''define the config file for voc and resnet101os8'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'voc',
    'rootdir': os.path.join(os.getcwd(), 'VOCdevkit/VOC2012'),
})
DATASET_CFG['train']['set'] = 'trainaug'
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'max_epochs': 60,
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update(
    {
        'num_classes': 21,
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'encnet_resnet101os8_voc_train',
        'logfilepath': 'encnet_resnet101os8_voc_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'encnet_resnet101os8_voc_test',
        'logfilepath': 'encnet_resnet101os8_voc_test/test.log',
        'resultsavepath': 'encnet_resnet101os8_voc_test/encnet_resnet101os8_voc_results.pkl'
    }
)