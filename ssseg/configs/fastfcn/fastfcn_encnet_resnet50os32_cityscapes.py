'''fastfcn_encnet_resnet50os32_cityscapes'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['FASTFCN_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_CITYSCAPES_512x1024'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS8'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 220
# modify other segmentor configs
SEGMENTOR_CFG.update({
    'benchmark': True,
    'num_classes': 19,
    'align_corners': False,
    'type': 'FastFCN',
    'segmentor': 'ENCNet',
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
        'pretrained': True, 'outstride': 32, 'use_conv3x3_stem': True, 'selected_indices': (1, 2, 3),
    },
    'head': {
        'jpu': {'in_channels_list': (512, 1024, 2048), 'mid_channels': 512, 'dilations': (1, 2, 4, 8)},
        'in_channels_list': [512, 1024, 2048], 'feats_channels': 512, 'num_codes': 32, 'dropout': 0.1,
        'extra': {'use_se_loss': True, 'add_lateral': False},
    },
    'auxiliary': {
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_se': {'type': 'CrossEntropyLoss', 'scale_factor': 0.2, 'reduction': 'mean', 'use_sigmoid': True},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
    }
})
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")