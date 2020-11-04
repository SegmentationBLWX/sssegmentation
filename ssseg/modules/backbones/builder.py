'''
Function:
    build the backbone
Author:
    Zhenchao Jin
'''
from .hrnet import HRNet
from .resnet import BuildResNet


'''build the backbone'''
def BuildBackbone(cfg):
    supported_backbones = {
        'hrnet': HRNet,
        'resnet': BuildResNet,
    }
    assert cfg['series'] in supported_backbones, 'unsupport backbone type %s...' % cfg['type']
    kwargs = {
        'outstride': cfg.get('outstride', 16),
        'pretrained': cfg.get('pretrained', True),
        'contract_dilation': cfg.get('contract_dilation', True),
        'pretrained_model_path': cfg.get('pretrained_model_path', ''),
        'is_improved_version': cfg.get('is_improved_version', True),
        'normlayer_opts': cfg.get('normlayer_opts', {'type': 'syncbatchnorm2d', 'opts': {}}),
    }
    return supported_backbones[cfg['series']](cfg['type'], **kwargs)