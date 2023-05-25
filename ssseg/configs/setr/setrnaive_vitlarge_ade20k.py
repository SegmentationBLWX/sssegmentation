'''setrnaive_vitlarge_ade20k'''
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
# modify other segmentor configs
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'jx_vit_large_p16_384', 'series': 'vit', 'img_size': (512, 512), 'drop_rate': 0., 'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels_list': [1024, 1024, 1024, 1024], 'feats_channels': 256, 'dropout': 0, 'num_convs': 2,
        'scale_factor': 4, 'kernel_size': 3, 'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'auxiliary': [
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 2, 'scale_factor': 4, 'kernel_size': 3},
    ],
})
SEGMENTOR_CFG['work_dir'] = 'setrnaive_vitlarge_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'setrnaive_vitlarge_ade20k/setrnaive_vitlarge_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'setrnaive_vitlarge_ade20k/setrnaive_vitlarge_ade20k_results.pkl'