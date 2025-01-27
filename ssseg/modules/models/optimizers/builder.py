'''
Function:
    Implementation of OptimizerBuilder and BuildOptimizer
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.optim as optim
from ...utils import BaseModuleBuilder
from .paramsconstructor import BuildParamsConstructor


'''OptimizerBuilder'''
class OptimizerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'SGD': optim.SGD, 'Adam': optim.Adam, 'AdamW': optim.AdamW, 'Adadelta': optim.Adadelta,
    }
    for optim_type in ['Adagrad', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop']:
        if hasattr(optim, optim_type):
            REGISTERED_MODULES[optim_type] = getattr(optim, optim_type)
    '''build'''
    def build(self, model_or_params, optimizer_cfg):
        # parse config
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optimizer_type = optimizer_cfg.pop('type')
        params_rules, filter_params = optimizer_cfg.pop('params_rules', {}), optimizer_cfg.pop('filter_params', False)
        # build params_constructor
        params_constructor = BuildParamsConstructor(params_rules=params_rules, filter_params=filter_params, optimizer_cfg=optimizer_cfg)
        # obtain params
        if isinstance(model_or_params, nn.Module):
            optimizer_cfg['params'] = params_constructor(model=model_or_params)
        else:
            optimizer_cfg['params'] = model_or_params
        # build optimizer
        optimizer = self.REGISTERED_MODULES[optimizer_type](**optimizer_cfg)
        # return
        return optimizer


'''BuildOptimizer'''
BuildOptimizer = OptimizerBuilder().build