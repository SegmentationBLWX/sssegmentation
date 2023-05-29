'''fcn_bisenetv1_resnet18os32_cityscapes'''
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
SEGMENTOR_CFG['losses'] = {
    'loss_aux1': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_aux2': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_cls': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
}
SEGMENTOR_CFG['work_dir'] = 'fcn_bisenetv1_resnet18os32_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'fcn_bisenetv1_resnet18os32_cityscapes/fcn_bisenetv1_resnet18os32_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'fcn_bisenetv1_resnet18os32_cityscapes/fcn_bisenetv1_resnet18os32_cityscapes_results.pkl'