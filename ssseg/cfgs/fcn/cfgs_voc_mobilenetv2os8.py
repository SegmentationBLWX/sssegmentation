'''define the config file for voc and mobilenetv2os8'''
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
        'backbone': {
            'type': 'mobilenetv2',
            'series': 'mobilenet',
            'pretrained': True,
            'outstride': 8,
            'selected_indices': (2, 3),
        },
        'decoder': {
            'in_channels': 320,
            'out_channels': 512,
            'dropout': 0.1,
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
        'backupdir': 'fcn_mobilenetv2os8_voc_train',
        'logfilepath': 'fcn_mobilenetv2os8_voc_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'fcn_mobilenetv2os8_voc_test',
        'logfilepath': 'fcn_mobilenetv2os8_voc_test/test.log',
        'resultsavepath': 'fcn_mobilenetv2os8_voc_test/fcn_mobilenetv2os8_voc_results.pkl'
    }
)