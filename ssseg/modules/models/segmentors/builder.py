'''
Function:
    Build the segmentor
Author:
    Zhenchao Jin
'''
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
from .pointrend import PointRend
from .deeplabv3 import Deeplabv3
from .lrasppnet import LRASPPNet
from .segformer import Segformer
from .memorynet import MemoryNet
from .setr import SETRUP, SETRMLA
from .maskformer import MaskFormer
from .semanticfpn import SemanticFPN
from .nonlocalnet import NonLocalNet
from .deeplabv3plus import Deeplabv3Plus
from .fcn import FCN, DepthwiseSeparableFCN


'''build segmentor'''
def BuildSegmentor(segmentor_cfg, mode, **kwargs):
    supported_segmentors = {
        'fcn': FCN,
        'ce2p': CE2P,
        'icnet': ICNet,
        'isnet': ISNet,
        'ccnet': CCNet,
        'danet': DANet,
        'gcnet': GCNet,
        'dmnet': DMNet,
        'isanet': ISANet,
        'encnet': ENCNet,
        'apcnet': APCNet,
        'emanet': EMANet,
        'pspnet': PSPNet,
        'psanet': PSANet,
        'ocrnet': OCRNet,
        'dnlnet': DNLNet,
        'annnet': ANNNet,
        'setrup': SETRUP,
        'setrmla': SETRMLA,
        'fastfcn': FastFCN,
        'upernet': UPerNet,
        'segformer': Segformer,
        'memorynet': MemoryNet,
        'pointrend': PointRend,
        'deeplabv3': Deeplabv3,
        'lrasppnet': LRASPPNet,
        'maskformer': MaskFormer,
        'semanticfpn': SemanticFPN,
        'nonlocalnet': NonLocalNet,
        'deeplabv3plus': Deeplabv3Plus,
        'depthwiseseparablefcn': DepthwiseSeparableFCN,
    }
    segmentor_type = segmentor_cfg['type']
    assert segmentor_type in supported_segmentors, 'unsupport segmentor_type %s...' % segmentor_type
    return supported_segmentors[segmentor_type](segmentor_cfg, mode=mode)