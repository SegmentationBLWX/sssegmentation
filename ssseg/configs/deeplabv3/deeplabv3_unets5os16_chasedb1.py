'''deeplabv3_unets5os16_chasedb1'''
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
    'type': None, 'series': 'unet', 'pretrained': False, 'selected_indices': (3, 4),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 64, 'feats_channels': 16, 'dilations': [1, 6, 12, 18], 'dropout': 0.1,
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
SEGMENTOR_CFG['work_dir'] = 'deeplabv3_unets5os16_chasedb1'
SEGMENTOR_CFG['logfilepath'] = 'deeplabv3_unets5os16_chasedb1/deeplabv3_unets5os16_chasedb1.log'
SEGMENTOR_CFG['resultsavepath'] = 'deeplabv3_unets5os16_chasedb1/deeplabv3_unets5os16_chasedb1_results.pkl'