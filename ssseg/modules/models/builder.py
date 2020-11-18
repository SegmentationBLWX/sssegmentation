'''
Function:
    builder for different models
Author:
    Zhenchao Jin
'''
from .fcn import FCN
from .ce2p import CE2P
from .pspnet import PSPNet
from .ocrnet import OCRNet
from .deeplabv3plus import Deeplabv3Plus


'''build model'''
def BuildModel(cfg, mode):
    supported_models = {
        'fcn': FCN,
        'ce2p': CE2P,
        'pspnet': PSPNet,
        'ocrnet': OCRNet,
        'deeplabv3plus': Deeplabv3Plus,
    }
    model_type = cfg['type']
    assert model_type in supported_models, 'unsupport model_type %s...' % model_type
    return supported_models[model_type](cfg, mode=mode)