'''upernet_convnextlarge21k_ade20k'''
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
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'convnext_large_21k', 'series': 'convnext', 'arch': 'large', 'pretrained': True, 'drop_path_rate': 0.4,
    'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
}
SEGMENTOR_CFG['head'] = {
    'in_channels_list': [192, 384, 768, 1536], 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 768, 'out_channels': 512, 'dropout': 0.1,
}
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (640, 640), 'stride': (426, 426)},
    'tricks': {
        'multiscale': [1], 'flip': False, 'use_probs_before_resize': False
    }
}
SEGMENTOR_CFG['work_dir'] = 'upernet_convnextlarge21k_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'upernet_convnextlarge21k_ade20k/upernet_convnextlarge21k_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'upernet_convnextlarge21k_ade20k/upernet_convnextlarge21k_ade20k_results.pkl'