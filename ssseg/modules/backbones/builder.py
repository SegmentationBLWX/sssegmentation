'''
Function:
    build the backbone
Author:
    Zhenchao Jin
'''
from .hrnet import BuildHRNet
from .resnet import BuildResNet
from .resnest import BuildResNeSt
from .mobilenet import BuildMobileNet
from .swin import BuildSwinTransformer
from .vit import BuildVisionTransformer


'''build the backbone'''
def BuildBackbone(cfg, **kwargs):
    supported_backbones = {
        'hrnet': BuildHRNet,
        'resnet': BuildResNet,
        'resnest': BuildResNeSt,
        'mobilenet': BuildMobileNet,
        'swin': BuildSwinTransformer,
        'vit': BuildVisionTransformer,
    }
    assert cfg['series'] in supported_backbones, 'unsupport backbone type %s...' % cfg['type']
    return supported_backbones[cfg['series']](cfg['type'], **cfg)