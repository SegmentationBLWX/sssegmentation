'''fastfcn_pspnet_resnet50os32_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS8


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS8.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 220
# modify other segmentor configs
SEGMENTOR_CFG.update({
    'benchmark': True,
    'num_classes': 19,
    'align_corners': False,
    'type': 'fastfcn',
    'segmentor': 'pspnet',
    'backend': 'nccl',
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'resnet50', 'series': 'resnet', 'pretrained': True, 
        'outstride': 32, 'use_stem': True, 'selected_indices': (1, 2, 3),
    },
    'head': {
        'jpu': {'in_channels_list': (512, 1024, 2048), 'mid_channels': 512, 'dilations': (1, 2, 4, 8)},
        'in_channels': 2048, 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    }
})
SEGMENTOR_CFG['work_dir'] = 'fastfcn_pspnet_resnet50os32_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'fastfcn_pspnet_resnet50os32_cityscapes/fastfcn_pspnet_resnet50os32_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'fastfcn_pspnet_resnet50os32_cityscapes/fastfcn_pspnet_resnet50os32_cityscapes_results.pkl'