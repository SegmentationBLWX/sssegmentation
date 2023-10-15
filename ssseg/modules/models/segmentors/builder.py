'''
Function:
    Build the segmentor
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
from .isanet import ISANet
from .encnet import ENCNet
from .apcnet import APCNet
from .emanet import EMANet
from .pspnet import PSPNet
from .psanet import PSANet
from .ocrnet import OCRNet
from .dnlnet import DNLNet
from .annnet import ANNNet
from .fastfcn import FastFCN
from .upernet import UPerNet
from .mobilesam import MobileSAM
from .pointrend import PointRend
from .deeplabv3 import Deeplabv3
from .lrasppnet import LRASPPNet
from .segformer import Segformer
from .memorynet import MemoryNet
from .setr import SETRUP, SETRMLA
from .maskformer import MaskFormer
from .memorynetv2 import MemoryNetV2
from .semanticfpn import SemanticFPN
from .nonlocalnet import NonLocalNet
from .deeplabv3plus import Deeplabv3Plus
from .fcn import FCN, DepthwiseSeparableFCN


'''BuildSegmentor'''
def BuildSegmentor(segmentor_cfg, mode):
    segmentor_cfg = copy.deepcopy(segmentor_cfg)
    # supported segmentors
    supported_segmentors = {
        'FCN': FCN, 'CE2P': CE2P, 'ICNet': ICNet, 'ISNet': ISNet, 'CCNet': CCNet, 'DANet': DANet,
        'GCNet': GCNet, 'DMNet': DMNet, 'ISANet': ISANet, 'ENCNet': ENCNet, 'APCNet': APCNet, 'SAM': SAM,
        'EMANet': EMANet, 'PSPNet': PSPNet, 'PSANet': PSANet, 'OCRNet': OCRNet, 'DNLNet': DNLNet,
        'ANNNet': ANNNet, 'SETRUP': SETRUP, 'SETRMLA': SETRMLA, 'FastFCN': FastFCN, 'UPerNet': UPerNet,
        'Segformer': Segformer, 'MemoryNet': MemoryNet, 'PointRend': PointRend, 'Deeplabv3': Deeplabv3,
        'LRASPPNet': LRASPPNet, 'MaskFormer': MaskFormer, 'MemoryNetV2': MemoryNetV2, 'SemanticFPN': SemanticFPN,
        'NonLocalNet': NonLocalNet, 'Deeplabv3Plus': Deeplabv3Plus, 'DepthwiseSeparableFCN': DepthwiseSeparableFCN,
        'MobileSAM': MobileSAM,
    }
    # build segmentor
    segmentor_type = segmentor_cfg.pop('type')
    segmentor = supported_segmentors[segmentor_type](cfg=segmentor_cfg, mode=mode)
    # return
    return segmentor