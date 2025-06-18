'''
Function:
    Implementation of Testing MiT
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.mit import DEFAULT_MODEL_URLS


'''MiTs'''
cfgs = [
{'type': 'MixVisionTransformer', 'structure_type': 'mit-b0', 'pretrained': True, 'pretrained_model_path': 'mit_b0.pth',
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
 'embed_dims': 32, 'num_stages': 4, 'num_layers': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
 'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,},
{'type': 'MixVisionTransformer', 'structure_type': 'mit-b1', 'pretrained': True, 'pretrained_model_path': 'mit_b1.pth',
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
 'embed_dims': 64, 'num_stages': 4, 'num_layers': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
 'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,},
{'type': 'MixVisionTransformer', 'structure_type': 'mit-b2', 'pretrained': True, 'pretrained_model_path': 'mit_b2.pth',
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
 'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 4, 6, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
 'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,},
{'type': 'MixVisionTransformer', 'structure_type': 'mit-b3', 'pretrained': True, 'pretrained_model_path': 'mit_b3.pth',
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
 'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 4, 18, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
 'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,},
{'type': 'MixVisionTransformer', 'structure_type': 'mit-b4', 'pretrained': True, 'pretrained_model_path': 'mit_b4.pth',
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
 'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 8, 27, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
 'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,},
{'type': 'MixVisionTransformer', 'structure_type': 'mit-b5', 'pretrained': True, 'pretrained_model_path': 'mit_b5.pth', 
 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
 'embed_dims': 64, 'num_stages': 4, 'num_layers': [3, 6, 40, 3], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
 'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,},
]
for cfg in cfgs:
    mit = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(
        structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict = mit.mitconvert(state_dict)
    try:
        mit.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        mit.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)