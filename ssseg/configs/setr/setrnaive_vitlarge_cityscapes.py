'''setrnaive_vitlarge_cityscapes'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['SETR_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_CITYSCAPES_768x768'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS8'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 220
# modify other segmentor configs
SEGMENTOR_CFG.update({
    'num_classes': 19,
    'backbone': {
        'type': 'VisionTransformer', 'structure_type': 'jx_vit_large_p16_384', 'img_size': (768, 768), 'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
        'patch_size': 16, 'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
        'qkv_bias': True, 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0., 'with_cls_token': True,
        'output_cls_token': False, 'patch_norm': False, 'final_norm': False, 'num_fcs': 2,
    },
    'head': {
        'in_channels_list': [1024, 1024, 1024, 1024], 'feats_channels': 256, 'dropout': 0, 'num_convs': 2,
        'scale_factor': 4, 'kernel_size': 3, 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
    },
    'auxiliary': [
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
    ],
})
SEGMENTOR_CFG['inference'] = {
    'forward': {'mode': 'slide', 'cropsize': (768, 768), 'stride': (512, 512)},
    'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': True},
    'evaluate': {'metric_list': ['iou', 'miou']},
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")