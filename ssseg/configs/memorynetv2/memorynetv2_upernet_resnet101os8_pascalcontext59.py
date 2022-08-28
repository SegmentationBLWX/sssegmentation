'''memorynetv2_upernet_resnet101os8_pascalcontext59'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'pascalcontext59',
    'rootdir': os.path.join(os.getcwd(), 'VOCdevkit/VOC2010/'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (480, 480), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (480, 480), 'data_type': 'tensor'}),
]
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = {
    'type': 'sgd',
    'lr': 0.004,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'params_rules': {'backbone_net': 0.1, 'others': 1.0},
}
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 260
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 59,
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
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# --single scale
INFERENCE_CFG.update({
    'mode': 'slide',
    'opts': {
        'cropsize': (480, 480),
        'stride': (320, 320),
    }, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': True,
    }
})
# --multi scale
'''
INFERENCE_CFG.update({
    'mode': 'slide',
    'opts': {
        'cropsize': (480, 480),
        'stride': (320, 320),
    }, 
    'tricks': {
        'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        'flip': True,
        'use_probs_before_resize': True,
    }
})
'''
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynetv2_upernet_resnet101os8_pascalcontext59'
COMMON_CFG['logfilepath'] = 'memorynetv2_upernet_resnet101os8_pascalcontext59/memorynetv2_upernet_resnet101os8_pascalcontext59.log'
COMMON_CFG['resultsavepath'] = 'memorynetv2_upernet_resnet101os8_pascalcontext59/memorynetv2_upernet_resnet101os8_pascalcontext59_results.pkl'