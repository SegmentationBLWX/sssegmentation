'''
Function:
    Implementation of SegmentorBuilder and BuildSegmentor
Author:
    Zhenchao Jin
'''
import copy
from .sam import SAM
from .ce2p import CE2P
from .icnet import ICNet
from .isnet import ISNet
from .ccnet import CCNet
from .danet import DANet
from .gcnet import GCNet
from .dmnet import DMNet
from .mcibi import MCIBI
from .samhq import SAMHQ
from .samv2 import SAMV2
from .idrnet import IDRNet
from .isanet import ISANet
from .encnet import ENCNet
from .apcnet import APCNet
from .emanet import EMANet
from .pspnet import PSPNet
from .psanet import PSANet
from .ocrnet import OCRNet
from .dnlnet import DNLNet
from .annnet import ANNNet
from .edgesam import EdgeSAM
from .fastfcn import FastFCN
from .upernet import UPerNet
from .mobilesam import MobileSAM
from .pointrend import PointRend
from .deeplabv3 import Deeplabv3
from .lrasppnet import LRASPPNet
from .segformer import Segformer
from .setr import SETRUP, SETRMLA
from .maskformer import MaskFormer
from .mask2former import Mask2Former
from .semanticfpn import SemanticFPN
from .nonlocalnet import NonLocalNet
from ...utils import BaseModuleBuilder
from .deeplabv3plus import Deeplabv3Plus
from .mcibiplusplus import MCIBIPlusPlus
from .fcn import FCN, DepthwiseSeparableFCN


'''SegmentorBuilder'''
class SegmentorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'FCN': FCN, 'CE2P': CE2P, 'ICNet': ICNet, 'ISNet': ISNet, 'CCNet': CCNet, 'DANet': DANet, 'SAMHQ': SAMHQ,
        'GCNet': GCNet, 'DMNet': DMNet, 'ISANet': ISANet, 'ENCNet': ENCNet, 'APCNet': APCNet, 'SAM': SAM, 'SAMV2': SAMV2,
        'EMANet': EMANet, 'PSPNet': PSPNet, 'PSANet': PSANet, 'OCRNet': OCRNet, 'DNLNet': DNLNet, 'Mask2Former': Mask2Former,
        'ANNNet': ANNNet, 'SETRUP': SETRUP, 'SETRMLA': SETRMLA, 'FastFCN': FastFCN, 'UPerNet': UPerNet, 'IDRNet': IDRNet, 
        'Segformer': Segformer, 'MCIBI': MCIBI, 'PointRend': PointRend, 'Deeplabv3': Deeplabv3, 'EdgeSAM': EdgeSAM,
        'LRASPPNet': LRASPPNet, 'MaskFormer': MaskFormer, 'MCIBIPlusPlus': MCIBIPlusPlus, 'SemanticFPN': SemanticFPN,
        'NonLocalNet': NonLocalNet, 'Deeplabv3Plus': Deeplabv3Plus, 'DepthwiseSeparableFCN': DepthwiseSeparableFCN,
        'MobileSAM': MobileSAM, 
    }
    '''build'''
    def build(self, segmentor_cfg, mode):
        segmentor_cfg = copy.deepcopy(segmentor_cfg)
        segmentor_type = segmentor_cfg.pop('type')
        segmentor = self.REGISTERED_MODULES[segmentor_type](cfg=segmentor_cfg, mode=mode)
        return segmentor


'''BuildSegmentor'''
BuildSegmentor = SegmentorBuilder().build