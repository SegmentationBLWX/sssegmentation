'''pspnet_unets5os16_hrf'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_HRF_256x256, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_HRF_256x256.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 1
# modify other segmentor configs
SEGMENTOR_CFG['type'] = 'PSPNet'
SEGMENTOR_CFG['num_classes'] = 2
SEGMENTOR_CFG['head'] = {
    'in_channels': 64, 'feats_channels': 16, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
}
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (256, 256), 'stride': (170, 170)}, 
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': True
    },
    'evaluate': {'metric_list': ['dice', 'mdice']},
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")