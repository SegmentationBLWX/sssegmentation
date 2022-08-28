'''memorynetv2_upernet_resnet101os8_vspw'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'vspw',
    'rootdir': os.path.join(os.getcwd(), 'VSPW_480p'),
})
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update({
    'params_rules': {'backbone_net': 0.1, 'others': 1.0},
})
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 240,
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 124,
    'backbone': {
        'type': 'resnet50',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
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
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
# modify inference config
INFERENCE_CFG = {
    'mode': 'slide',
    'opts': {'cropsize': (512, 512), 'stride': (341, 341)}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': True
    }
}
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynetv2_upernet_resnet101os8_vspw'
COMMON_CFG['logfilepath'] = 'memorynetv2_upernet_resnet101os8_vspw/memorynetv2_upernet_resnet101os8_vspw.log'
COMMON_CFG['resultsavepath'] = 'memorynetv2_upernet_resnet101os8_vspw/memorynetv2_upernet_resnet101os8_vspw_results.pkl'