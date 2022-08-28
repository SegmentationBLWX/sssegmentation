'''memorynetv2_upernet_resnet101os8_ade20k'''
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
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 130
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
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
# --single-scale
INFERENCE_CFG = INFERENCE_CFG.copy()
# --multi-scale
'''
INFERENCE_CFG = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        'flip': True,
        'use_probs_before_resize': True
    }
}
'''
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynetv2_upernet_resnet101os8_ade20k'
COMMON_CFG['logfilepath'] = 'memorynetv2_upernet_resnet101os8_ade20k/memorynetv2_upernet_resnet101os8_ade20k.log'
COMMON_CFG['resultsavepath'] = 'memorynetv2_upernet_resnet101os8_ade20k/memorynetv2_upernet_resnet101os8_ade20k_results.pkl'