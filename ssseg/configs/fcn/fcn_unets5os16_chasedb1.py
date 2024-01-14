'''fcn_unets5os16_chasedb1'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CHASEDB1_128x128, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CHASEDB1_128x128.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 1
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 2
SEGMENTOR_CFG['backbone'] = {
    'type': 'UNet', 'structure_type': 'unets5os16', 'pretrained': False, 'selected_indices': (3, 4),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 64, 'feats_channels': 64, 'dropout': 0.1,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 128, 'out_channels': 64, 'dropout': 0.1,
}
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (128, 128), 'stride': (85, 85)}, 
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': True
    },
    'metric_list': ['dice', 'mdice'],
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")