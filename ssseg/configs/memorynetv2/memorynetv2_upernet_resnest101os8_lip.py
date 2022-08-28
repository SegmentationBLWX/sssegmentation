'''memorynetv2_upernet_resnest101os8_lip'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'lip',
    'rootdir': os.path.join(os.getcwd(), 'LIP'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': (0.75, 1.25)}),
    ('RandomCrop', {'crop_size': (473, 473), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5, 'fix_ann_pairs': [(15, 14), (17, 16), (19, 18)]}),
    ('RandomRotation', {'angle_upper': 30, 'rotation_prob': 0.6}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (473, 473), 'data_type': 'tensor'}),
]
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (473, 473), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
DATALOADER_CFG['train'].update({
    'batch_size': 32,
})
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 150
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 20,
    'backbone': {
        'type': 'resnest101',
        'series': 'resnest',
        'pretrained': True,
        'outstride': 8,
        'selected_indices': (0, 1, 2, 3),
    },
})
SEGMENTOR_CFG['head']['fpn'] = {
    'in_channels_list': [256, 512, 1024, 2048],
    'feats_channels': 1024,
    'out_channels': 512,
}
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cls': {'in_channels': 2560, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
}
SEGMENTOR_CFG['head']['context_within_image']['type'] = 'ppm'
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
# modify inference config
# --single-scale
INFERENCE_CFG = INFERENCE_CFG.copy()
# --single-scale with flipping
'''
INFERENCE_CFG = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [1],
        'flip': True,
        'use_probs_before_resize': False
    }
}
'''
# --multi-scale with flipping
'''
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
INFERENCE_CFG = {
    'mode': 'slide',
    'opts': {'cropsize': (473, 473), 'stride': (315, 315)}, 
    'tricks': {
        'multiscale': [0.75, 1, 1.25],
        'flip': True,
        'use_probs_before_resize': True
    }
}
'''
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynetv2_upernet_resnest101os8_lip'
COMMON_CFG['logfilepath'] = 'memorynetv2_upernet_resnest101os8_lip/memorynetv2_upernet_resnest101os8_lip.log'
COMMON_CFG['resultsavepath'] = 'memorynetv2_upernet_resnest101os8_lip/memorynetv2_upernet_resnest101os8_lip_results.pkl'