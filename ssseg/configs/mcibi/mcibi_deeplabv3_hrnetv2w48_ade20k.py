'''mcibi_deeplabv3_hrnetv2w48_ade20k'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['MCIBI_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_ADE20k_512x512'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 180
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.004, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'HRNet', 'structure_type': 'hrnetv2_w48', 'arch': 'hrnetv2_w48', 'pretrained': True, 'selected_indices': (0, 0, 0, 0),
}
SEGMENTOR_CFG['head']['in_channels'] = sum([48, 96, 192, 384])
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.004
SEGMENTOR_CFG['auxiliary'] = None
SEGMENTOR_CFG['losses'].pop('loss_aux')
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")


# modify inference config
# --single-scale
SEGMENTOR_CFG['inference'] = SEGMENTOR_CFG['inference'].copy()
# --multi-scale with flipping
'''
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (512, 512), 'stride': (341, 341)},
    'tta': {'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75], 'flip': True, 'use_probs_before_resize': True},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
'''