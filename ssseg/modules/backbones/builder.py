'''
Function:
    build the backbone
Author:
    Zhenchao Jin
'''
from .hrnet import BuildHRNet
from .resnet import BuildResNet
from .mobilenet import BuildMobileNet


'''build the backbone'''
def BuildBackbone(cfg):
    supported_backbones = {
        'hrnet': BuildHRNet,
        'resnet': BuildResNet,
        'mobilenet': BuildMobileNet,
    }
    assert cfg['series'] in supported_backbones, 'unsupport backbone type %s...' % cfg['type']
    supported_keys = ['outstride', 'pretrained', 'out_indices', 'contract_dilation'
                      'pretrained_model_path', 'is_improved_version', 'normlayer_opts']
    kwargs = {}
    for key in supported_keys:
        if key in cfg: kwargs.update({key: cfg[key]})
    return supported_backbones[cfg['series']](cfg['type'], **kwargs)