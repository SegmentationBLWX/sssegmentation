'''
Function:
    Implementation of ActivationBuilder and BuildActivation
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from .swish import Swish
from .hardswish import HardSwish
from .hardsigmoid import HardSigmoid
from .....utils import BaseModuleBuilder


'''ActivationBuilder'''
class ActivationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid, 'Swish': Swish,
    }
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU', 
                     'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            REGISTERED_MODULES[act_type] = getattr(nn, act_type)
    '''build'''
    def build(self, act_cfg):
        if act_cfg is None: return nn.Identity()
        return super().build(act_cfg)


'''BuildActivation'''
BuildActivation = ActivationBuilder().build