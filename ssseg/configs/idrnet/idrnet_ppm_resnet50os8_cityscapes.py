'''idrnet_ppm_resnet50os8_cityscapes'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS8


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS8.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 220
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
SEGMENTOR_CFG['head']['use_fpn_before'] = False
SEGMENTOR_CFG['head']['use_fpn_after'] = True
SEGMENTOR_CFG['head']['use_sa_on_coarsecontext_before'] = False
SEGMENTOR_CFG['head']['use_sa_on_coarsecontext_after'] = True
SEGMENTOR_CFG['head']['force_stage1_use_oripr'] = True
SEGMENTOR_CFG['head']['feats_channels'] = 1024
SEGMENTOR_CFG['head']['coarse_context'] = {'type': 'ppm', 'pool_scales': [1, 2, 3, 6]}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")