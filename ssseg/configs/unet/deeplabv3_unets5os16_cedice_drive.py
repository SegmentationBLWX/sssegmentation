'''deeplabv3_unets5os16_cedice_drive'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['UNET_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_DRIVE_64x64'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 1
# modify other segmentor configs
SEGMENTOR_CFG['type'] = 'Deeplabv3'
SEGMENTOR_CFG['num_classes'] = 2
SEGMENTOR_CFG['head'] = {
    'in_channels': 64, 'feats_channels': 16, 'dilations': [1, 6, 12, 18], 'dropout': 0.1,
}
SEGMENTOR_CFG['losses'] = {
    'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
    'loss_cls': [
        {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
        {'type': 'DiceLoss', 'scale_factor': 3.0, 'ignore_index': 255, 'reduction': 'mean'},
    ],
}
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (64, 64), 'stride': (42, 42)},
    'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': True},
    'evaluate': {'metric_list': ['dice', 'mdice']},
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")