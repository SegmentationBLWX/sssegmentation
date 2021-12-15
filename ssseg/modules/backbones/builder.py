'''
Function:
    Build the backbone network
Author:
    Zhenchao Jin
'''
from .unet import BuildUNet
from .cgnet import BuildCGNet
from .hrnet import BuildHRNet
from .erfnet import BuildERFNet
from .resnet import BuildResNet
from .resnest import BuildResNeSt
from .fastscnn import BuildFastSCNN
from .bisenetv1 import BuildBiSeNetV1
from .bisenetv2 import BuildBiSeNetV2
from .mobilenet import BuildMobileNet
from .swin import BuildSwinTransformer
from .vit import BuildVisionTransformer
from .mit import BuildMixVisionTransformer
from .timmwrapper import BuildTIMMBackbone


'''build the backbone'''
def BuildBackbone(cfg, **kwargs):
    supported_backbones = {
        'unet': BuildUNet,
        'cgnet': BuildCGNet,
        'hrnet': BuildHRNet,
        'erfnet': BuildERFNet,
        'resnet': BuildResNet,
        'resnest': BuildResNeSt,
        'fastscnn': BuildFastSCNN,
        'timm': BuildTIMMBackbone,
        'bisenetv1': BuildBiSeNetV1,
        'bisenetv2': BuildBiSeNetV2,
        'mobilenet': BuildMobileNet,
        'swin': BuildSwinTransformer,
        'vit': BuildVisionTransformer,
        'mit': BuildMixVisionTransformer,
    }
    assert cfg['series'] in supported_backbones, 'unsupport backbone type %s...' % cfg['type']
    return supported_backbones[cfg['series']](cfg['type'], **cfg)