'''
Function:
    Implementation of Testing ConvNeXt
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.convnext import DEFAULT_MODEL_URLS


'''ConvNeXts'''
configs = {
    'convnext_tiny': {
        'type': 'ConvNeXt', 'structure_type': 'convnext_tiny', 'arch': 'tiny', 'pretrained': True, 'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
    'convnext_small': {
        'type': 'ConvNeXt', 'structure_type': 'convnext_small', 'arch': 'small', 'pretrained': True, 'drop_path_rate': 0.3,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
    'convnext_base': {
        'type': 'ConvNeXt', 'structure_type': 'convnext_base', 'arch': 'base', 'pretrained': True, 'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
    'convnext_base_21k': {
        'type': 'ConvNeXt', 'structure_type': 'convnext_base_21k', 'arch': 'base', 'pretrained': True, 'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
    'convnext_large_21k': {
        'type': 'ConvNeXt', 'structure_type': 'convnext_large_21k', 'arch': 'large', 'pretrained': True, 'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
    'convnext_xlarge_21k': {
        'type': 'ConvNeXt', 'structure_type': 'convnext_xlarge_21k', 'arch': 'xlarge', 'pretrained': True, 'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
}
for name, cfg in configs.items():
    convnext = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(
        structure_type=name, pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict_convert = {}
    for key, value in state_dict.items():
        state_dict_convert[key.replace('backbone.', '')] = value
    try:
        convnext.load_state_dict(state_dict_convert, strict=False)
    except Exception as err:
        print(err)
    try:
        convnext.load_state_dict(state_dict_convert, strict=True)
    except Exception as err:
        print(err)