'''
Function:
    Implementation of Testing SwinTransformer
Author:
    Zhenchao Jin
'''
import torch.nn.functional as F
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.swin import DEFAULT_MODEL_URLS


'''SwinTransformers'''
cfgs = [
{'type': 'SwinTransformer', 'structure_type': 'swin_large_patch4_window12_384_22kto1k', 'pretrained': True, 
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 192, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_large_patch4_window12_384_22k', 'pretrained': True, 
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 192, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_base_patch4_window12_384', 'pretrained': True,
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 128, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_base_patch4_window7_224', 'pretrained': True,
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 224, 'in_channels': 3, 'embed_dims': 128, 'patch_size': 4, 'window_size': 7, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_base_patch4_window12_384_22k', 'pretrained': True,
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 128, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_base_patch4_window7_224_22k', 'pretrained': True,
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 224, 'in_channels': 3, 'embed_dims': 128, 'patch_size': 4, 'window_size': 7, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_small_patch4_window7_224', 'pretrained': True, 
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 224, 'in_channels': 3, 'embed_dims': 96, 'patch_size': 4, 'window_size': 7, 'mlp_ratio': 4,
 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
{'type': 'SwinTransformer', 'structure_type': 'swin_tiny_patch4_window7_224', 'pretrained': True, 
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'pretrain_img_size': 224, 'in_channels': 3, 'embed_dims': 96, 'patch_size': 4, 'window_size': 7, 'mlp_ratio': 4, 
 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True, 
 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,},
]
for cfg in cfgs:
    swin = BuildBackbone(cfg)
    state_dict = loadpretrainedweights(
        structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict = swin.swinconvert(state_dict)
    # be consistent
    from collections import OrderedDict
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            state_dict_new[k[9:]] = v
        else:
            state_dict_new[k] = v
    state_dict = state_dict_new
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = swin.absolute_pos_embed.size()
        if not (N1 != N2 or C1 != C2 or L != H * W):
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
    # interpolate position bias table if needed
    relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = swin.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if (nH1 == nH2) and (L1 != L2):
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()
    try:
        swin.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        swin.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)