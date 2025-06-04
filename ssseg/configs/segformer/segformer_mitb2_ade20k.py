'''segformer_mitb2_ade20k'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['SEGFORMER_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_ADE20k_512x512'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'MixVisionTransformer', 'structure_type': 'mit-b2', 'pretrained': True, 'pretrained_model_path': 'mit_b2.pth',
    'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
    'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 4, 6, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
    'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
}
SEGMENTOR_CFG['head'] = {
    'in_channels_list': [64, 128, 320, 512], 'feats_channels': 256, 'dropout': 0.1,
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")