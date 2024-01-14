'''fcn_bisenetv1_resnet18os32_cityscapes'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_1024x1024, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_1024x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 860
SEGMENTOR_CFG['scheduler']['min_lr'] = 1e-4
SEGMENTOR_CFG['scheduler']['warmup_cfg'] = {'type': 'linear', 'ratio': 0.1, 'iters': 1000}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'BiSeNetV1', 'structure_type': 'bisenetv1', 'pretrained': False, 'selected_indices': (0, 1, 2),
    'spatial_channels_list': (64, 64, 64, 128), 'context_channels_list': (128, 256, 512), 'out_channels': 256,
    'backbone_cfg': {'type': 'ResNet', 'structure_type': 'resnet18conv3x3stem', 'depth': 18, 'pretrained': True, 'outstride': 32, 'use_conv3x3_stem': True},
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 256, 'feats_channels': 256, 'dropout': 0.1, 'num_convs': 1,
}
SEGMENTOR_CFG['auxiliary'] = [
    {'in_channels': 128, 'out_channels': 64, 'dropout': 0.1, 'num_convs': 1},
    {'in_channels': 128, 'out_channels': 64, 'dropout': 0.1, 'num_convs': 1},
]
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")