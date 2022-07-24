'''memorynet_deeplabv3_vitlarge_cocostuff10k'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cocostuff10k',
    'rootdir': os.path.join(os.getcwd(), 'COCOStuff10k'),
})
DATASET_CFG['test']['set'] = 'test'
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = {
    'type': 'sgd',
    'sgd': {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0,
    },
    'max_epochs': 110,
    'params_rules': {'backbone_net': 0.1, 'others': 1.0},
    'policy': {
        'type': 'poly',
        'opts': {'power': 0.9, 'max_iters': None, 'num_iters': None, 'num_epochs': None}
    },
    'adjust_period': ['iteration', 'epoch'][0],
}
# modify losses config
LOSSES_CFG = {
    'loss_aux1': {
        'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_aux2': {
        'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_aux3': {
        'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_cls_stage1': {
        'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_cls_stage2': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
}
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 182,
    'backbone': {
        'type': 'jx_vit_large_p16_384',
        'series': 'vit',
        'img_size': (473, 473),
        'drop_rate': 0.,
        'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
        'pretrained': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'normlayer': {
        'in_channels_list': [1024, 1024, 1024, 1024],
        'type': 'layernorm', 
        'eps': 1e-6,
    },
    'auxiliary': [
        {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
        {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
        {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
    ],
})
SEGMENTOR_CFG['memory']['use_loss'] = False
SEGMENTOR_CFG['memory']['in_channels'] = 1024
SEGMENTOR_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.001 * 0.9
SEGMENTOR_CFG['memory']['context_within_image']['cfg']['dilations'] = [1, 6, 12, 18]
# modify inference config
INFERENCE_CFG = {
    'mode': 'slide',
    'opts': {'cropsize': (512, 512), 'stride': (341, 341)}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': True
    }
}
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynet_deeplabv3_vitlarge_cocostuff10k'
COMMON_CFG['logfilepath'] = 'memorynet_deeplabv3_vitlarge_cocostuff10k/memorynet_deeplabv3_vitlarge_cocostuff10k.log'
COMMON_CFG['resultsavepath'] = 'memorynet_deeplabv3_vitlarge_cocostuff10k/memorynet_deeplabv3_vitlarge_cocostuff10k_results.pkl'