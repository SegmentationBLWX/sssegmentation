'''memorynet_deeplabv3_vitlarge_cocostuff10k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_COCOStuff10k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_COCOStuff10k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
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
SEGMENTOR_CFG['work_dir'] = 'memorynet_deeplabv3_vitlarge_cocostuff10k'
SEGMENTOR_CFG['logfilepath'] = 'memorynet_deeplabv3_vitlarge_cocostuff10k/memorynet_deeplabv3_vitlarge_cocostuff10k.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynet_deeplabv3_vitlarge_cocostuff10k/memorynet_deeplabv3_vitlarge_cocostuff10k_results.pkl'