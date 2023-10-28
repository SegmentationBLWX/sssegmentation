'''
Function:
    Implementation of OptimizerBuilder and BuildOptimizer
Author:
    Zhenchao Jin
'''
import copy
import torch.optim as optim
from ...utils import BaseModuleBuilder
from .paramsconstructor import BuildParamsConstructor


'''OptimizerBuilder'''
class OptimizerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'SGD': optim.SGD, 'Adam': optim.Adam, 'AdamW': optim.AdamW, 'Adadelta': optim.Adadelta,
    }
    '''build'''
    def build(self, model, optimizer_cfg):
        # parse config
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optimizer_type = optimizer_cfg.pop('type')
        params_rules, filter_params = optimizer_cfg.pop('params_rules', {}), optimizer_cfg.pop('filter_params', False)
        # build params_constructor
        params_constructor = BuildParamsConstructor(params_rules=params_rules, filter_params=filter_params, optimizer_cfg=optimizer_cfg)
        # obtain params
        optimizer_cfg['params'] = params_constructor(model=model)
        # build optimizer
        optimizer = self.REGISTERED_MODULES[optimizer_type](**optimizer_cfg)
        # return
        return optimizer


'''BuildOptimizer'''
BuildOptimizer = OptimizerBuilder().build