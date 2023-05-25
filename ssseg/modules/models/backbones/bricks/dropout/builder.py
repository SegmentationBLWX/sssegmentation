'''
Function:
    Implementation of BuildDropout
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from .droppath import DropPath


'''BuildDropout'''
def BuildDropout(dropout_cfg):
    if dropout_cfg is None: return nn.Identity()
    dropout_cfg = copy.deepcopy(dropout_cfg)
    # supported dropouts
    supported_dropouts = {
        'DropPath': DropPath,
        'Dropout': nn.Dropout,
        'Dropout2d': nn.Dropout2d,
        'Dropout3d': nn.Dropout3d,
    }
    # build dropout
    dropout_type = dropout_cfg.pop('type')
    dropout = supported_dropouts[dropout_type](**dropout_cfg)
    # return
    return dropout