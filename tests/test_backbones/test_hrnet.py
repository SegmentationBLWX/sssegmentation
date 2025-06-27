'''
Function:
    Implementation of Testing HRNet
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.hrnet import DEFAULT_MODEL_URLS


'''hrnets'''
cfgs = [
    {'type': 'HRNet', 'structure_type': 'hrnetv2_w18_small', 'arch': 'hrnetv2_w18_small', 'pretrained': True, 'selected_indices': (0, 0),},
    {'type': 'HRNet', 'structure_type': 'hrnetv2_w18', 'arch': 'hrnetv2_w18', 'pretrained': True, 'selected_indices': (0, 0),},
    {'type': 'HRNet', 'structure_type': 'hrnetv2_w32', 'arch': 'hrnetv2_w32', 'pretrained': True, 'selected_indices': (0, 0),},
    {'type': 'HRNet', 'structure_type': 'hrnetv2_w40', 'arch': 'hrnetv2_w40', 'pretrained': True, 'selected_indices': (0, 0),},
    {'type': 'HRNet', 'structure_type': 'hrnetv2_w48', 'arch': 'hrnetv2_w48', 'pretrained': True, 'selected_indices': (0, 0),},
]
for cfg in cfgs:
    hrnet = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS)
    try:
        hrnet.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        hrnet.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)