'''mcibiplusplus_resnet50os8_cocostuff10k'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_COCOStuff10k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_COCOStuff10k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 110
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 182
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['losses'].pop('loss_cwi')
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")