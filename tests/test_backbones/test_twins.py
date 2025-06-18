'''
Function:
    Implementation of Testing Twins
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.twins import DEFAULT_MODEL_URLS


'''Twins'''
cfgs = [
{'type': 'PCPVT', 'structure_type': 'pcpvt_base', 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
 'norm_cfg': {'type': 'LayerNorm'}, 'depths': [3, 4, 18, 3], 'drop_path_rate': 0.3,},
{'type': 'PCPVT', 'structure_type': 'pcpvt_large', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 
 'norm_cfg': {'type': 'LayerNorm'}, 'depths': [3, 8, 27, 3], 'drop_path_rate': 0.3,},
{'type': 'PCPVT', 'structure_type': 'pcpvt_small', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 
 'norm_cfg': {'type': 'LayerNorm'}, 'depths': [3, 4, 6, 3], 'drop_path_rate': 0.2,},
{'type': 'SVT', 'structure_type': 'svt_base', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'embed_dims': [96, 192, 384, 768], 'num_heads': [3, 6, 12, 24], 'mlp_ratios': [4, 4, 4, 4], 'depths': [2, 2, 18, 2], 
 'windiow_sizes': [7, 7, 7, 7], 'norm_after_stage': True, 'drop_path_rate': 0.2},
{'type': 'SVT', 'structure_type': 'svt_large', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'embed_dims': [128, 256, 512, 1024], 'num_heads': [4, 8, 16, 32], 'mlp_ratios': [4, 4, 4, 4], 'depths': [2, 2, 18, 2], 
 'windiow_sizes': [7, 7, 7, 7], 'norm_after_stage': True, 'drop_path_rate': 0.3},
{'type': 'SVT', 'structure_type': 'svt_small', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
 'embed_dims': [64, 128, 256, 512], 'num_heads': [2, 4, 8, 16], 'mlp_ratios': [4, 4, 4, 4], 'depths': [2, 2, 10, 4], 
 'windiow_sizes': [7, 7, 7, 7], 'norm_after_stage': True, 'drop_path_rate': 0.2},
]
for cfg in cfgs:
    twins = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(
        structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict = twins.twinsconvert(cfg['structure_type'], state_dict)
    try:
        twins.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        twins.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)