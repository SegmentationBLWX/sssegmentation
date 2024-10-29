'''mcibiplusplus_aspp_resnest101os8_cityscapes'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
SEGMENTOR_CFG['dataset']['train']['set'] = 'trainval'
SEGMENTOR_CFG['dataset']['test']['set'] = 'test'
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 440
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.004, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNeSt', 'depth': 101, 'structure_type': 'resnest101', 'pretrained': True, 'outstride': 8, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cls': {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
}
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (1024, 512), 'stride': (341, 341)},
    'tta': {'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75], 'flip': True, 'use_probs_before_resize': True},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")