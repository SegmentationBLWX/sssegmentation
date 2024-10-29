'''upernet_convnextbase21k_ade20k'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_640x640, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_640x640.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
SEGMENTOR_CFG['scheduler']['min_lr'] = 0.0
SEGMENTOR_CFG['scheduler']['power'] = 1.0
SEGMENTOR_CFG['scheduler']['warmup_cfg'] = {'type': 'linear', 'ratio': 1e-6, 'iters': 1500}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'ConvNeXt', 'structure_type': 'convnext_base_21k', 'arch': 'base', 'pretrained': True, 'drop_path_rate': 0.4,
    'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
}
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (640, 640), 'stride': (426, 426)},
    'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")