'''
Function:
    Implementation of BuildActivation
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from .swish import Swish
from .hardswish import HardSwish
from .hardsigmoid import HardSigmoid 


'''BuildActivation'''
def BuildActivation(act_cfg):
    if act_cfg is None: return nn.Identity()
    act_cfg = copy.deepcopy(act_cfg)
    # supported activations
    supported_activations = {
        'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid, 'Swish': Swish,
    }
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU',
                     'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            supported_activations[act_type] = getattr(nn, act_type)
    # build activation
    act_type = act_cfg.pop('type')
    activation = supported_activations[act_type](**act_cfg)
    # return
    return activation