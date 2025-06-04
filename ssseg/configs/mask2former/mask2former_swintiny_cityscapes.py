'''mask2former_swintiny_cityscapes'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['MASK2FORMER_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_CITYSCAPES_512x1024'].copy()
SEGMENTOR_CFG['dataset']['train']['data_pipelines'][0] = (
    'RandomChoiceResize', {
        'scales': [int(1024 * x * 0.1) for x in range(5, 21)], 
        'resize_type': 'ResizeShortestEdge', 
        'max_size': 4096,
    }
)
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 500
SEGMENTOR_CFG['scheduler']['min_lr'] = 0.0
SEGMENTOR_CFG['scheduler']['clipgrad_cfg'] = {'max_norm': 0.01, 'norm_type': 2}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'SwinTransformer', 'structure_type': 'swin_tiny_patch4_window7_224', 'pretrained': True, 
    'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
    'pretrain_img_size': 224, 'in_channels': 3, 'embed_dims': 96, 'patch_size': 4, 'window_size': 7, 'mlp_ratio': 4, 
    'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True, 
    'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,
}
SEGMENTOR_CFG['head']['pixel_decoder']['input_shape']['in_channels'] = [96, 192, 384, 768]
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")
# modify training tricks in scheduler config
SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'] = {
    'base_setting': dict(norm_wd_multiplier=0.0),
    'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0),
    'backbone_net.patch_embed.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0),
    'backbone_net.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0),
    'absolute_pos_embed': dict(lr_multiplier=0.1, wd_multiplier=0.0),
    'relative_position_bias_table': dict(lr_multiplier=0.1, wd_multiplier=0.0),
    'query_embed': dict(lr_multiplier=1.0, wd_multiplier=0.0),
    'query_feat': dict(lr_multiplier=1.0, wd_multiplier=0.0),
    'level_embed': dict(lr_multiplier=1.0, wd_multiplier=0.0),
}
for stage_id, num_blocks in enumerate(SEGMENTOR_CFG['backbone']['depths']):
    for block_id in range(num_blocks):
        SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'].update({
            f'backbone_net.stages.{stage_id}.blocks.{block_id}.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0)
        })
for stage_id in range(len(SEGMENTOR_CFG['backbone']['depths']) - 1):
    SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'].update({
        f'backbone_net.stages.{stage_id}.downsample.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0)
    })