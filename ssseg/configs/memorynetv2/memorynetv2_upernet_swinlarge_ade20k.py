'''memorynetv2_upernet_swinlarge_ade20k'''
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
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'AdamW', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
    'params_rules': {'backbone_net_zerowd': (1.0, 0.0), 'others': (1.0, 1.0)},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
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
    'cls': {'in_channels': 2560, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
}
SEGMENTOR_CFG['head']['in_channels'] = 1536
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = True
SEGMENTOR_CFG['auxiliary'] = {'in_channels': 768, 'out_channels': 512, 'dropout': 0.1}
SEGMENTOR_CFG['work_dir'] = 'memorynetv2_upernet_swinlarge_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'memorynetv2_upernet_swinlarge_ade20k/memorynetv2_upernet_swinlarge_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynetv2_upernet_swinlarge_ade20k/memorynetv2_upernet_swinlarge_ade20k_results.pkl'


# modify inference config
# --single-scale
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (640, 640), 'stride': (426, 426)}, 
    'tricks': {
        'multiscale': [1.0],
        'flip': False,
        'use_probs_before_resize': True
    }
}
# --multi-scale
'''
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (640, 640), 'stride': (426, 426)}, 
    'tricks': {
        'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        'flip': True,
        'use_probs_before_resize': True
    }
}
'''