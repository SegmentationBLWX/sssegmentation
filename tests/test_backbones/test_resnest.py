'''
Function:
    Implementation of Testing ResNeSt
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.resnest import DEFAULT_MODEL_URLS


'''resnests'''
for depth in [50, 101, 200]:
    resnest = BuildBackbone(backbone_cfg={
        'type': 'ResNeSt', 'depth': depth, 'structure_type': f'resnest{depth}', 'pretrained': True, 'outstride': 8, 'selected_indices': (0, 1, 2, 3), 'stem_channels': {50: 64, 101: 128, 200: 128}[depth]
    })
    state_dict = loadpretrainedweights(
        structure_type=f'resnest{depth}', pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    try:
        resnest.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        resnest.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)