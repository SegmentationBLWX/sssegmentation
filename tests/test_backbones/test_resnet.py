'''
Function:
    Implementation of Testing ResNet
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.resnet import DEFAULT_MODEL_URLS


'''resnets'''
for depth in [18, 34, 50, 101, 152]:
    resnet = BuildBackbone(backbone_cfg={
        'type': 'ResNet', 'depth': depth, 'structure_type': f'resnet{depth}',
        'pretrained': False, 'outstride': 8, 'use_conv3x3_stem': False, 'selected_indices': (2, 3),
    })
    state_dict = loadpretrainedweights(
        structure_type=f'resnet{depth}', pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    try:
        resnet.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        resnet.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)


'''resnets with replaced stem'''
for depth in [18, 50, 101]:
    resnet = BuildBackbone(backbone_cfg={
        'type': 'ResNet', 'depth': depth, 'structure_type': f'resnet{depth}conv3x3stem',
        'pretrained': False, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
    })
    state_dict = loadpretrainedweights(
        structure_type=f'resnet{depth}conv3x3stem', pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    try:
        resnet.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        resnet.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)