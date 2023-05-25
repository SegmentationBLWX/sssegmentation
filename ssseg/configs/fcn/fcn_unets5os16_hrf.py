'''fcn_unets5os16_hrf'''
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
SEGMENTOR_CFG['num_classes'] = 2
SEGMENTOR_CFG['backbone'] = {
    'type': None, 'series': 'unet', 'pretrained': False, 'selected_indices': (3, 4),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 64, 'feats_channels': 64, 'dropout': 0.1,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 128, 'out_channels': 64, 'dropout': 0.1,
}
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (256, 256), 'stride': (170, 170)}, 
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': True
    },
    'metric_list': ['dice', 'mdice'],
}
SEGMENTOR_CFG['work_dir'] = 'fcn_unets5os16_hrf'
SEGMENTOR_CFG['logfilepath'] = 'fcn_unets5os16_hrf/fcn_unets5os16_hrf.log'
SEGMENTOR_CFG['resultsavepath'] = 'fcn_unets5os16_hrf/fcn_unets5os16_hrf_results.pkl'