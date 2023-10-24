'''
Function:
    Implementation of OptimizerBuilder and BuildOptimizer
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.optim as optim
from .paramsconstructor import DefaultParamsConstructor, LayerDecayParamsConstructor


'''OptimizerBuilder'''
class OptimizerBuilder():
    REGISTERED_OPTIMIZERS = {
        'SGD': optim.SGD, 'Adam': optim.Adam, 'AdamW': optim.AdamW, 'Adadelta': optim.Adadelta,
    }
    REGISTERED_PARAMS_CONSTRUCTORS = {
        'DefaultParamsConstructor': DefaultParamsConstructor, 'LayerDecayParamsConstructor': LayerDecayParamsConstructor,
    }
    def __init__(self, require_register_optimizers=None, require_update_optimizers=None, require_register_params_constructors=None, require_update_params_constructors=None):
        # register params optimizers
        if require_register_optimizers and isinstance(require_register_optimizers, dict):
            for optimizer_type, optimizer in require_register_optimizers.items():
                self.register(optimizer_type, optimizer, 'optimizer')
        if require_update_optimizers and isinstance(require_update_optimizers, dict):
            for optimizer_type, optimizer in require_update_optimizers.items():
                self.update(optimizer_type, optimizer, 'optimizer')
        # register params constructor
        if require_register_params_constructors and isinstance(require_register_params_constructors, dict):
            for params_constructor_type, params_constructor in require_register_params_constructors.items():
                self.register(params_constructor_type, params_constructor, 'params_constructor')
        if require_update_params_constructors and isinstance(require_update_params_constructors, dict):
            for params_constructor_type, params_constructor in require_update_params_constructors.items():
                self.update(params_constructor_type, params_constructor, 'params_constructor')
    '''build'''
    def build(self, model, optimizer_cfg):
        # parse config
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optimizer_type = optimizer_cfg.pop('type')
        params_rules, filter_params = {}, False
        if 'params_rules' in optimizer_cfg:
            params_rules = optimizer_cfg.pop('params_rules')
        if 'filter_params' in optimizer_cfg:
            filter_params = optimizer_cfg.pop('filter_params')
        # build params_constructor
        constructor_type = params_rules.get('type', 'DefaultParamsConstructor')
        params_constructor = self.REGISTERED_PARAMS_CONSTRUCTORS[constructor_type](params_rules=params_rules, filter_params=filter_params, optimizer_cfg=optimizer_cfg)
        # obtain params
        optimizer_cfg['params'] = params_constructor(model=model)
        # build optimizer
        optimizer = self.REGISTERED_OPTIMIZERS[optimizer_type](**optimizer_cfg)
        # return
        return optimizer
    '''register'''
    def register(self, module_type, module, mode='params_constructor'):
        assert mode in ['optimizer', 'params_constructor']
        if mode == 'optimizer':
            assert module_type not in self.REGISTERED_OPTIMIZERS
            self.REGISTERED_OPTIMIZERS[module_type] = module
        else:
            assert module_type not in self.REGISTERED_PARAMS_CONSTRUCTORS
            self.REGISTERED_PARAMS_CONSTRUCTORS[module_type] = module
    '''update'''
    def update(self, module_type, module, mode='params_constructor'):
        assert mode in ['optimizer', 'params_constructor']
        if mode == 'optimizer':
            assert module_type in self.REGISTERED_OPTIMIZERS
            self.REGISTERED_OPTIMIZERS[module_type] = module
        else:
            assert module_type in self.REGISTERED_PARAMS_CONSTRUCTORS
            self.REGISTERED_PARAMS_CONSTRUCTORS[module_type] = module


'''BuildOptimizer'''
BuildOptimizer = OptimizerBuilder().build