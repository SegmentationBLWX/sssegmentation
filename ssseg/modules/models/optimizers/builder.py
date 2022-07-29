'''
Funtion:
    Build the optimizer
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.optim as optim
from .paramsconstructor import DefaultParamsConstructor, LayerDecayParamsConstructor


'''BuildOptimizer'''
def BuildOptimizer(model, optimizer_cfg):
    # define the supported optimizers
    supported_optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adadelta': optim.Adadelta,
    }
    # parse optimizer_cfg
    optimizer_cfg = copy.deepcopy(optimizer_cfg)
    optimizer_type = optimizer_cfg.pop('type')
    params_rules, filter_params = {}, False
    if 'params_rules' in optimizer_cfg:
        params_rules = optimizer_cfg.pop('params_rules')
    if 'filter_params' in optimizer_cfg:
        filter_params = optimizer_cfg.pop('filter_params')
    # obtain params
    supported_constructors = {
        'default': DefaultParamsConstructor,
        'layerdecay': LayerDecayParamsConstructor,
    }
    constructor_type = params_rules.get('type', 'default')
    params_constructor = supported_constructors[constructor_type](params_rules=params_rules, filter_params=filter_params, optimizer_cfg=optimizer_cfg)
    optimizer_cfg['params'] = params_constructor(model=model)
    # return
    return supported_optimizers[optimizer_type](**optimizer_cfg)