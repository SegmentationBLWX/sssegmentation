'''memorynet_deeplabv3_vitlarge_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0, 'params_rules': {'backbone_net': 0.1, 'others': 1.0},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'jx_vit_large_p16_384', 'series': 'vit', 'img_size': (512, 512), 'drop_rate': 0., 'out_indices': (9, 14, 19, 23),
    'norm_cfg': {'type': 'layernorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head']['in_channels'] = 1024
SEGMENTOR_CFG['head']['context_within_image']['cfg']['dilations'] = [1, 6, 12, 18]
SEGMENTOR_CFG['head']['norm_cfg'] = {'in_channels_list': [1024, 1024, 1024, 1024], 'type': 'layernorm', 'eps': 1e-6}
SEGMENTOR_CFG['auxiliary'] = [
    {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
    {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
    {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
]
SEGMENTOR_CFG['losses'] = {
    'loss_aux1': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_aux2': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_aux3': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_cls_stage1': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_cls_stage2': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
}
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (512, 512), 'stride': (341, 341)}, 
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': True
    }
}
SEGMENTOR_CFG['work_dir'] = 'memorynet_deeplabv3_vitlarge_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'memorynet_deeplabv3_vitlarge_ade20k/memorynet_deeplabv3_vitlarge_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynet_deeplabv3_vitlarge_ade20k/memorynet_deeplabv3_vitlarge_ade20k_results.pkl'