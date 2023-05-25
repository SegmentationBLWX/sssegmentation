'''
Function:
    Implementation of BuildActivation
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from .hardswish import HardSwish
from .hardsigmoid import HardSigmoid 


'''BuildActivation'''
def BuildActivation(act_cfg):
    if act_cfg is None: return nn.Identity()
    act_cfg = copy.deepcopy(act_cfg)
    # supported activations
    supported_activations = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'ReLU6': nn.ReLU6,
        'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid,
        'HardSwish': HardSwish,
        'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid,
    }
    # build activation
    act_type = act_cfg.pop('type')
    activation = supported_activations[act_type](**act_cfg)
    # return
    return activation