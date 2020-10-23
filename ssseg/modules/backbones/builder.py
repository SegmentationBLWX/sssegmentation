'''
Function:
    build the backbone
Author:
    Zhenchao Jin
'''
from .resnet import BuildResNet


'''build the backbone'''
def BuildBackbone(cfg):
    supported_backbones = {
        'resnet': BuildResNet,
    }
    assert cfg['series'] in supported_backbones, 'unsupport backbone type %s...' % cfg['type']
    kwargs = {
        'outstride': cfg.get('outstride', 16),
        'pretrained': cfg.get('pretrained', True),
        'contract_dilation': cfg.get('contract_dilation', True),
        'pretrainedmodelpath': cfg.get('pretrainedmodelpath', ''),
        'is_improved_version': cfg.get('is_improved_version', False),
        'normlayer_opts': cfg.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}}),
    }
    return supported_backbones[cfg['series']](cfg['type'], **kwargs)