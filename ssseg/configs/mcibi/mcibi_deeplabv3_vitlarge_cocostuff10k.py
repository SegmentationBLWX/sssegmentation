'''mcibi_deeplabv3_vitlarge_cocostuff10k'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['MCIBI_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_COCOStuff10k_512x512'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 110
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0, 
    'params_rules': {
        'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0),
    },
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 182
SEGMENTOR_CFG['backbone'] = {
    'type': 'VisionTransformer', 'structure_type': 'jx_vit_large_p16_384', 'img_size': (512, 512), 'out_indices': (9, 14, 19, 23),
    'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
    'patch_size': 16, 'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
    'qkv_bias': True, 'drop_rate': 0.1, 'attn_drop_rate': 0., 'drop_path_rate': 0., 'with_cls_token': True,
    'output_cls_token': False, 'patch_norm': False, 'final_norm': False, 'num_fcs': 2,
}
SEGMENTOR_CFG['head']['use_loss'] = False
SEGMENTOR_CFG['head']['in_channels'] = 1024
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.001 * 0.9
SEGMENTOR_CFG['head']['context_within_image']['cfg']['dilations'] = [1, 12, 24, 36]
SEGMENTOR_CFG['head']['norm_cfg'] = {'in_channels_list': [1024, 1024, 1024, 1024], 'type': 'LayerNorm', 'eps': 1e-6}
SEGMENTOR_CFG['auxiliary'] = [
    {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
    {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
    {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'num_convs': 2, 'upsample': {'scale_factor': 4}},
]
SEGMENTOR_CFG['losses'] = {
    'loss_aux1': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
    'loss_aux2': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
    'loss_aux3': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
    'loss_cls_stage1': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
    'loss_cls_stage2': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
}
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (512, 512), 'stride': (341, 341)},
    'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': True},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
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