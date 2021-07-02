'''
Function:
    builder for different models
Author:
    Zhenchao Jin
'''
from .fcn import FCN
from .ce2p import CE2P
from .ccnet import CCNet
from .danet import DANet
from .gcnet import GCNet
from .dmnet import DMNet
from .setr import SETRUP
from .encnet import ENCNet
from .apcnet import APCNet
from .emanet import EMANet
from .pspnet import PSPNet
from .psanet import PSANet
from .ocrnet import OCRNet
from .dnlnet import DNLNet
from .annnet import ANNNet
from .upernet import UPerNet
from .deeplabv3 import Deeplabv3
from .lrasppnet import LRASPPNet
from .semanticfpn import SemanticFPN
from .nonlocalnet import NonLocalNet
from .deeplabv3plus import Deeplabv3Plus


'''build model'''
def BuildModel(cfg, mode, **kwargs):
    supported_models = {
        'fcn': FCN,
        'ce2p': CE2P,
        'ccnet': CCNet,
        'danet': DANet,
        'gcnet': GCNet,
        'dmnet': DMNet,
        'encnet': ENCNet,
        'apcnet': APCNet,
        'emanet': EMANet,
        'pspnet': PSPNet,
        'psanet': PSANet,
        'ocrnet': OCRNet,
        'dnlnet': DNLNet,
        'annnet': ANNNet,
        'setrup': SETRUP,
        'upernet': UPerNet,
        'deeplabv3': Deeplabv3,
        'lrasppnet': LRASPPNet,
        'semanticfpn': SemanticFPN,
        'nonlocalnet': NonLocalNet,
        'deeplabv3plus': Deeplabv3Plus,
    }
    model_type = cfg['type']
    assert model_type in supported_models, 'unsupport model_type %s...' % model_type
    return supported_models[model_type](cfg, mode=mode)