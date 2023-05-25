'''fcn_bisenetv2fp16_cityscapes'''
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
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.05, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': None, 'series': 'bisenetv2', 'pretrained': False, 'selected_indices': (0, 1, 2, 3, 4),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 128, 'feats_channels': 1024, 'dropout': 0.1, 'num_convs': 1,
}
SEGMENTOR_CFG['auxiliary'] = [
    {'in_channels': 16, 'out_channels': 16, 'dropout': 0.1, 'num_convs': 2},
    {'in_channels': 32, 'out_channels': 64, 'dropout': 0.1, 'num_convs': 2},
    {'in_channels': 64, 'out_channels': 256, 'dropout': 0.1, 'num_convs': 2},
    {'in_channels': 128, 'out_channels': 1024, 'dropout': 0.1, 'num_convs': 2},
]
SEGMENTOR_CFG['losses'] = {
    'loss_aux1': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_aux2': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_aux3': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_aux4': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_cls': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
}
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'apex', 'opt_level': 'O1'}
SEGMENTOR_CFG['work_dir'] = 'fcn_bisenetv2fp16_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'fcn_bisenetv2fp16_cityscapes/fcn_bisenetv2fp16_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'fcn_bisenetv2fp16_cityscapes/fcn_bisenetv2fp16_cityscapes_results.pkl'