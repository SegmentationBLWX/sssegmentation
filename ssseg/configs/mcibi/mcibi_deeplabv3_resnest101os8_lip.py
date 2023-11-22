'''mcibi_deeplabv3_resnest101os8_lip'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_LIP_473x473, DATALOADER_CFG_BS40


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_LIP_473x473.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS40.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 150
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.007, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 20
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNeSt', 'depth': 101, 'structure_type': 'resnest101', 'pretrained': True, 'outstride': 8, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head']['use_loss'] = False
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.007
SEGMENTOR_CFG['work_dir'] = 'mcibi_deeplabv3_resnest101os8_lip'
SEGMENTOR_CFG['logfilepath'] = 'mcibi_deeplabv3_resnest101os8_lip/mcibi_deeplabv3_resnest101os8_lip.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibi_deeplabv3_resnest101os8_lip/mcibi_deeplabv3_resnest101os8_lip_results.pkl'


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