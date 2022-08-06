'''
Function:
    Build the backbone network
Author:
    Zhenchao Jin
'''
import copy
from .unet import BuildUNet
from .beit import BuildBEiT
from .twins import BuildTwins
from .cgnet import BuildCGNet
from .hrnet import BuildHRNet
from .erfnet import BuildERFNet
from .resnet import BuildResNet
from .resnest import BuildResNeSt
from .fastscnn import BuildFastSCNN
from .convnext import BuildConvNeXt
from .bisenetv1 import BuildBiSeNetV1
from .bisenetv2 import BuildBiSeNetV2
from .mobilenet import BuildMobileNet
from .swin import BuildSwinTransformer
from .vit import BuildVisionTransformer
from .mit import BuildMixVisionTransformer
from .timmwrapper import BuildTIMMBackbone


'''BuildBackbone'''
def BuildBackbone(backbone_cfg):
    supported_backbones = {
        'unet': BuildUNet,
        'beit': BuildBEiT,
        'twins': BuildTwins,
        'cgnet': BuildCGNet,
        'hrnet': BuildHRNet,
        'erfnet': BuildERFNet,
        'resnet': BuildResNet,
        'resnest': BuildResNeSt,
        'fastscnn': BuildFastSCNN,
        'convnext': BuildConvNeXt,
        'timm': BuildTIMMBackbone,
        'bisenetv1': BuildBiSeNetV1,
        'bisenetv2': BuildBiSeNetV2,
        'mobilenet': BuildMobileNet,
        'swin': BuildSwinTransformer,
        'vit': BuildVisionTransformer,
        'mit': BuildMixVisionTransformer,
    }
    selected_backbone = supported_backbones[backbone_cfg['series']]
    backbone_cfg = copy.deepcopy(backbone_cfg)
    backbone_cfg.pop('series')
    return selected_backbone(backbone_cfg)