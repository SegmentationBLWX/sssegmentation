'''upernet_beitlarge_ade20k'''
import os
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
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'AdamW', 'lr': 3e-5, 'betas': (0.9, 0.999), 'weight_decay': 0.05,
    'params_rules': {'type': 'LearningRateDecayParamsConstructor', 'num_layers': 24, 'decay_rate': 0.95, 'decay_type': 'layer_wise_vit'},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'BEiT', 'structure_type': 'beit_large_patch16_224_pt22k_ft22k', 'pretrained': True, 
    'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
    'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
    'qv_bias': True, 'init_values': 1e-6, 'drop_path_rate': 0.2, 'out_indices': [7, 11, 15, 23]
}
SEGMENTOR_CFG['head'] = {
    'feature2pyramid': {'embed_dim': 1024, 'rescales': [4, 2, 1, 0.5]}, 'in_channels_list': [1024, 1024, 1024, 1024],
    'feats_channels': 1024, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")