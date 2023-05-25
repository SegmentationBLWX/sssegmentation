'''
Function:
    Implementation of BuildBackbone
Author:
    Zhenchao Jin
'''
import copy
from .unet import UNet
from .beit import BEiT
from .cgnet import CGNet
from .hrnet import HRNet
from .erfnet import ERFNet
from .resnet import ResNet
from .resnest import ResNeSt
from .twins import PCPVT, SVT
from .fastscnn import FastSCNN
from .convnext import ConvNeXt
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .swin import SwinTransformer
from .vit import VisionTransformer
from .mit import MixVisionTransformer
from .timmwrapper import TIMMBackbone
from .mobilenet import MobileNetV2, MobileNetV3


'''BuildBackbone'''
def BuildBackbone(backbone_cfg):
    backbone_cfg = copy.deepcopy(backbone_cfg)
    # supported backbones
    supported_backbones = {
        'UNet': UNet, 'BEiT': BEiT, 'CGNet': CGNet, 'HRNet': HRNet,
        'ERFNet': ERFNet, 'ResNet': ResNet, 'ResNeSt': ResNeSt, 'PCPVT': PCPVT,
        'SVT': SVT, 'FastSCNN': FastSCNN, 'ConvNeXt': ConvNeXt, 'BiSeNetV1': BiSeNetV1,
        'BiSeNetV2': BiSeNetV2, 'SwinTransformer': SwinTransformer, 'VisionTransformer': VisionTransformer,
        'MixVisionTransformer': MixVisionTransformer, 'TIMMBackbone': TIMMBackbone, 
        'MobileNetV2': MobileNetV2, 'MobileNetV3': MobileNetV3,
    }
    # build backbone
    backbone_type = backbone_cfg.pop('type')
    backbone = supported_backbones[backbone_type](**backbone_cfg)
    # return
    return backbone