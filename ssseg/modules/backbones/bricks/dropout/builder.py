'''
Function:
    build dropout
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .droppath import DropPath


'''build dropout'''
def BuildDropout(dropout_type, **kwargs):
    supported_dropouts = {
        'droppath': DropPath,
        'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
    }
    assert dropout_type in supported_dropouts, 'unsupport dropout type %s...' % dropout_type
    return supported_dropouts[dropout_type](**kwargs)