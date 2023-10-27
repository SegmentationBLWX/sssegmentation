'''memorynetv2_upernet_resnest101os8_pascalcontext59'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_PASCALCONTEXT59_480x480, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_PASCALCONTEXT59_480x480.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 260
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.004, 'momentum': 0.9, 'weight_decay': 1e-4, 'params_rules': {'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0)},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 59
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
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (480, 480), 'stride': (320, 320)}, 
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': True,
    }
}
SEGMENTOR_CFG['work_dir'] = 'memorynetv2_upernet_resnest101os8_pascalcontext59'
SEGMENTOR_CFG['logfilepath'] = 'memorynetv2_upernet_resnest101os8_pascalcontext59/memorynetv2_upernet_resnest101os8_pascalcontext59.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynetv2_upernet_resnest101os8_pascalcontext59/memorynetv2_upernet_resnest101os8_pascalcontext59_results.pkl'


# modify inference config
# --single scale
SEGMENTOR_CFG['inference'] = SEGMENTOR_CFG['inference'].copy()
# --multi scale
'''
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (480, 480), 'stride': (320, 320)}, 
    'tricks': {
        'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        'flip': True,
        'use_probs_before_resize': True,
    }
}
'''