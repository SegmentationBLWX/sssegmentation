'''pspnet_unets5os16_drive'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_DRIVE_64x64, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_DRIVE_64x64.copy()
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
    'in_channels': 64, 'feats_channels': 16, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 128, 'out_channels': 64, 'dropout': 0.1,
}
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (64, 64), 'stride': (42, 42)}, 
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': True
    },
    'metric_list': ['dice', 'mdice'],
}
SEGMENTOR_CFG['work_dir'] = 'pspnet_unets5os16_drive'
SEGMENTOR_CFG['logfilepath'] = 'pspnet_unets5os16_drive/pspnet_unets5os16_drive.log'
SEGMENTOR_CFG['resultsavepath'] = 'pspnet_unets5os16_drive/pspnet_unets5os16_drive_results.pkl'