'''
Function:
    build activation functions
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .hardswish import HardSwish
from .hardsigmoid import HardSigmoid 


'''build activation functions'''
def BuildActivation(activation_type, **kwargs):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
        'hardswish': HardSwish,
        'identity': nn.Identity,
        'leakyrelu': nn.LeakyReLU,
        'hardsigmoid': HardSigmoid,
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type](**kwargs)