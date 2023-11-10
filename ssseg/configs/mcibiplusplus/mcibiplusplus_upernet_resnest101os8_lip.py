'''mcibiplusplus_upernet_resnest101os8_lip'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_LIP_473x473, DATALOADER_CFG_BS32


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_LIP_473x473.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS32.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 150
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 20
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNeSt', 'depth': 101, 'structure_type': 'resnest101', 'pretrained': True, 'outstride': 8, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head']['fpn'] = {
    'in_channels_list': [256, 512, 1024, 2048], 'feats_channels': 1024, 'out_channels': 512,
}
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cls': {'in_channels': 2560, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
}
SEGMENTOR_CFG['head']['context_within_image']['type'] = 'ppm'
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
SEGMENTOR_CFG['work_dir'] = 'mcibiplusplus_upernet_resnest101os8_lip'
SEGMENTOR_CFG['logfilepath'] = 'mcibiplusplus_upernet_resnest101os8_lip/mcibiplusplus_upernet_resnest101os8_lip.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibiplusplus_upernet_resnest101os8_lip/mcibiplusplus_upernet_resnest101os8_lip_results.pkl'


# modify inference config
# --single-scale
SEGMENTOR_CFG['inference'] = SEGMENTOR_CFG['inference'].copy()
# --single-scale with flipping
'''
SEGMENTOR_CFG['inference'] = {
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
SEGMENTOR_CFG['dataset']['test']['data_pipelines'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (473, 473), 'stride': (315, 315)}, 
    'tricks': {
        'multiscale': [0.75, 1, 1.25],
        'flip': True,
        'use_probs_before_resize': True
    }
}
'''