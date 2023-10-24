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
from .mask2former import Mask2Former
from .semanticfpn import SemanticFPN
from .nonlocalnet import NonLocalNet
from .deeplabv3plus import Deeplabv3Plus
from .fcn import FCN, DepthwiseSeparableFCN


'''SegmentorBuilder'''
class SegmentorBuilder():
    REGISTERED_SEGMENTORS = {
        'FCN': FCN, 'CE2P': CE2P, 'ICNet': ICNet, 'ISNet': ISNet, 'CCNet': CCNet, 'DANet': DANet,
        'GCNet': GCNet, 'DMNet': DMNet, 'ISANet': ISANet, 'ENCNet': ENCNet, 'APCNet': APCNet, 'SAM': SAM,
        'EMANet': EMANet, 'PSPNet': PSPNet, 'PSANet': PSANet, 'OCRNet': OCRNet, 'DNLNet': DNLNet,
        'ANNNet': ANNNet, 'SETRUP': SETRUP, 'SETRMLA': SETRMLA, 'FastFCN': FastFCN, 'UPerNet': UPerNet,
        'Segformer': Segformer, 'MemoryNet': MemoryNet, 'PointRend': PointRend, 'Deeplabv3': Deeplabv3,
        'LRASPPNet': LRASPPNet, 'MaskFormer': MaskFormer, 'MemoryNetV2': MemoryNetV2, 'SemanticFPN': SemanticFPN,
        'NonLocalNet': NonLocalNet, 'Deeplabv3Plus': Deeplabv3Plus, 'DepthwiseSeparableFCN': DepthwiseSeparableFCN,
        'MobileSAM': MobileSAM, 'IDRNet': IDRNet, 'Mask2Former': Mask2Former,
    }
    def __init__(self, require_register_segmentors=None, require_update_segmentors=None):
        if require_register_segmentors and isinstance(require_register_segmentors, dict):
            for segmentor_type, segmentor in require_register_segmentors.items():
                self.register(segmentor_type, segmentor)
        if require_update_segmentors and isinstance(require_update_segmentors, dict):
            for segmentor_type, segmentor in require_update_segmentors.items():
                self.update(segmentor_type, segmentor)
    '''build'''
    def build(self, segmentor_cfg, mode):
        segmentor_cfg = copy.deepcopy(segmentor_cfg)
        segmentor_type = segmentor_cfg.pop('type')
        segmentor = self.REGISTERED_SEGMENTORS[segmentor_type](cfg=segmentor_cfg, mode=mode)
        return segmentor
    '''register'''
    def register(self, segmentor_type, segmentor):
        assert segmentor_type not in self.REGISTERED_SEGMENTORS
        self.REGISTERED_SEGMENTORS[segmentor_type] = segmentor
    '''update'''
    def update(self, segmentor_type, segmentor):
        assert segmentor_type in self.REGISTERED_SEGMENTORS
        self.REGISTERED_SEGMENTORS[segmentor_type] = segmentor


'''BuildSegmentor'''
BuildSegmentor = SegmentorBuilder().build