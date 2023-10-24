'''
Function:
    Implementation of ActivationBuilder and BuildActivation
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from .swish import Swish
from .hardswish import HardSwish
from .hardsigmoid import HardSigmoid 


'''ActivationBuilder'''
class ActivationBuilder():
    REGISTERED_ACTIVATIONS = {
        'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid, 'Swish': Swish,
    }
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU', 
                     'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            REGISTERED_ACTIVATIONS[act_type] = getattr(nn, act_type)
    def __init__(self, require_register_activations=None, require_update_activations=None):
        if require_register_activations and isinstance(require_register_activations, dict):
            for act_type, act_module in require_register_activations.items():
                self.register(act_type, act_module)
        if require_update_activations and isinstance(require_update_activations, dict):
            for act_type, act_module in require_update_activations.items():
                self.update(act_type, act_module)
    '''build'''
    def build(self, act_cfg):
        if act_cfg is None: return nn.Identity()
        act_cfg = copy.deepcopy(act_cfg)
        act_type = act_cfg.pop('type')
        activation = self.REGISTERED_ACTIVATIONS[act_type](**act_cfg)
        return activation
    '''register'''
    def register(self, act_type, act_module):
        assert act_type not in self.REGISTERED_ACTIVATIONS
        self.REGISTERED_ACTIVATIONS[act_type] = act_module
    '''update'''
    def update(self, act_type, act_module):
        assert act_type in self.REGISTERED_ACTIVATIONS
        self.REGISTERED_ACTIVATIONS[act_type] = act_module


'''BuildActivation'''
BuildActivation = ActivationBuilder().build