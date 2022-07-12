'''
Function:
    Build dropout
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from .droppath import DropPath


'''BuildDropout'''
def BuildDropout(dropout_cfg):
    supported_dropouts = {
        'droppath': DropPath,
        'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
    }
    selected_dropout_func = supported_dropouts[dropout_cfg['type']]
    dropout_cfg = copy.deepcopy(dropout_cfg)
    dropout_cfg.pop('type')
    return selected_dropout_func(**dropout_cfg)