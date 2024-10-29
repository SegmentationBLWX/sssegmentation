'''mcibiplusplus_upernet_swinlarge_lip'''
import os
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_LIP_473x473, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_LIP_473x473.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 110
SEGMENTOR_CFG['scheduler']['min_lr'] = 0.0
SEGMENTOR_CFG['scheduler']['power'] = 1.0
SEGMENTOR_CFG['scheduler']['warmup_cfg'] = {'type': 'linear', 'ratio': 1e-6, 'iters': 1500}
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'AdamW', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
    'params_rules': {
        'absolute_pos_embed': dict(wd_multiplier=0.),
        'relative_position_bias_table': dict(wd_multiplier=0.),
        'norm': dict(wd_multiplier=0.),
    },
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 20
SEGMENTOR_CFG['backbone'] = {
    'type': 'SwinTransformer', 'structure_type': 'swin_large_patch4_window12_384_22k', 'pretrained': True, 
    'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
    'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 192, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
    'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
    'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,
}
SEGMENTOR_CFG['head']['fpn'] = {
    'in_channels_list': [192, 384, 768, 1024], 'feats_channels': 1024, 'out_channels': 512,
}
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cls': {'in_channels': 2560, 'out_channels': 512, 'dropout': 0.1},
}
SEGMENTOR_CFG['head']['in_channels'] = 1536
SEGMENTOR_CFG['head']['context_within_image']['type'] = 'ppm'
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
SEGMENTOR_CFG['auxiliary'] = {'in_channels': 768, 'out_channels': 512, 'dropout': 0.1}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['evaluate_results_filename'] = f"{os.path.split(__file__)[-1].split('.')[0]}.pkl"
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")


# modify inference config
# --single-scale
SEGMENTOR_CFG['inference'] = SEGMENTOR_CFG['inference'].copy()
# --single-scale with flipping
'''
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
    'tta': {'multiscale': [1], 'flip': True, 'use_probs_before_resize': False},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
'''
# --multi-scale
'''
SEGMENTOR_CFG['dataset']['test']['data_pipelines'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (473, 473), 'stride': (315, 315)},
    'tta': {'multiscale': [0.75, 1, 1.25], 'flip': True, 'use_probs_before_resize': True},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
'''