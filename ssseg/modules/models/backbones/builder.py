'''
Function:
    Implementation of BackboneBuilder and BuildBackbone
Author:
    Zhenchao Jin
'''
import copy
from .mae import MAE
from .unet import UNet
from .beit import BEiT
from .cgnet import CGNet
from .hrnet import HRNet
from .erfnet import ERFNet
from .resnet import ResNet
from .samvit import SAMViT
from .resnest import ResNeSt
from .twins import PCPVT, SVT
from .fastscnn import FastSCNN
from .convnext import ConvNeXt
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .swin import SwinTransformer
from .convnextv2 import ConvNeXtV2
from .vit import VisionTransformer
from .mit import MixVisionTransformer
from .timmwrapper import TIMMBackbone
from .hiera import Hiera, HieraWithFPN
from ...utils import BaseModuleBuilder
from .edgesamrepvit import EdgeSAMRepViT
from .mobilevit import MobileViT, MobileViTV2
from .mobilesamtinyvit import MobileSAMTinyViT
from .mobilenet import MobileNetV2, MobileNetV3


'''BackboneBuilder'''
class BackboneBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'UNet': UNet, 'BEiT': BEiT, 'CGNet': CGNet, 'HRNet': HRNet, 'MobileViT': MobileViT, 'MobileViTV2': MobileViTV2,
        'ERFNet': ERFNet, 'ResNet': ResNet, 'ResNeSt': ResNeSt, 'PCPVT': PCPVT, 'MobileSAMTinyViT': MobileSAMTinyViT, 
        'SVT': SVT, 'FastSCNN': FastSCNN, 'ConvNeXt': ConvNeXt, 'BiSeNetV1': BiSeNetV1, 'MAE': MAE, 'SAMViT': SAMViT,
        'SwinTransformer': SwinTransformer, 'VisionTransformer': VisionTransformer, 'EdgeSAMRepViT': EdgeSAMRepViT,
        'MixVisionTransformer': MixVisionTransformer, 'TIMMBackbone': TIMMBackbone, 'ConvNeXtV2': ConvNeXtV2, 'Hiera': Hiera,
        'MobileNetV2': MobileNetV2, 'MobileNetV3': MobileNetV3, 'BiSeNetV2': BiSeNetV2, 'HieraWithFPN': HieraWithFPN,
    }
    '''build'''
    def build(self, backbone_cfg):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if 'selected_indices' in backbone_cfg: backbone_cfg.pop('selected_indices')
        return super().build(backbone_cfg)


'''BuildBackbone'''
BuildBackbone = BackboneBuilder().build