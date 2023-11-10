'''mcibiplusplus_upernet_resnest101os8_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 180
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.004, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
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
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
SEGMENTOR_CFG['work_dir'] = 'mcibiplusplus_upernet_resnest101os8_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'mcibiplusplus_upernet_resnest101os8_ade20k/mcibiplusplus_upernet_resnest101os8_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibiplusplus_upernet_resnest101os8_ade20k/mcibiplusplus_upernet_resnest101os8_ade20k_results.pkl'


# modify inference config
# --single-scale
SEGMENTOR_CFG['inference'] = SEGMENTOR_CFG['inference'].copy()
# --multi-scale
'''
SEGMENTOR_CFG['inference'] = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [0.75, 1.0, 1.5],
        'flip': True,
        'use_probs_before_resize': True
    }
}
'''