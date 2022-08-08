'''memorynet_deeplabv3_hrnetv2w48_cityscapes'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cityscapes',
    'rootdir': os.path.join(os.getcwd(), 'CityScapes'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),
]
DATASET_CFG['train']['set'] = 'trainval'
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 500,
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
LOSSES_CFG.pop('loss_aux')
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 19,
    'backbone': {
        'type': 'hrnetv2_w48',
        'series': 'hrnet',
        'pretrained': True,
        'selected_indices': (0, 0, 0, 0),
    },
    'auxiliary': None,
})
SEGMENTOR_CFG['head']['use_loss'] = False
SEGMENTOR_CFG['head']['downsample_backbone']['stride'] = 2
SEGMENTOR_CFG['head']['in_channels'] = sum([48, 96, 192, 384])
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.01 * 0.9
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynet_deeplabv3_hrnetv2w48_cityscapes'
COMMON_CFG['logfilepath'] = 'memorynet_deeplabv3_hrnetv2w48_cityscapes/memorynet_deeplabv3_hrnetv2w48_cityscapes.log'
COMMON_CFG['resultsavepath'] = 'memorynet_deeplabv3_hrnetv2w48_cityscapes/memorynet_deeplabv3_hrnetv2w48_cityscapes_results.pkl'