'''
Function:
    Implementation of Testing Mobilenets
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.mobilenet import DEFAULT_MODEL_URLS


'''mobilenetv2'''
cfgs = [
    {'type': 'MobileNetV2', 'structure_type': 'mobilenetv2', 'pretrained': True, 'outstride': 8, 'selected_indices': (0, 1, 2, 3),},
]
for cfg in cfgs:
    mobilenet = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS)
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith('backbone.'):
            value = state_dict.pop(key)
            key = '.'.join(key.split('.')[1:])
            state_dict[key] = value
    try:
        mobilenet.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        mobilenet.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)


'''mobilenetv3'''
cfgs = [
    {'type': 'MobileNetV3', 'structure_type': 'mobilenetv3_small', 'pretrained': True, 'outstride': 8,
     'arch_type': 'small', 'out_indices': (0, 1, 12), 'selected_indices': (0, 1, 2),},
    {'type': 'MobileNetV3', 'structure_type': 'mobilenetv3_large', 'pretrained': True,
     'outstride': 8, 'arch_type': 'large', 'selected_indices': (0, 1, 2),},
]
for cfg in cfgs:
    mobilenet = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS)
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith('backbone.'):
            value = state_dict.pop(key)
            key = '.'.join(key.split('.')[1:])
            state_dict[key] = value
    try:
        mobilenet.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        mobilenet.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)