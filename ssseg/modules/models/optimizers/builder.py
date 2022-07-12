'''
Funtion:
    Build the optimizer
Author:
    Zhenchao Jin
'''
from .sgd import BuildSGD
from .adam import BuildAdam
from .adamw import BuildAdamW


'''BuildOptimizer'''
def BuildOptimizer(model, optimizer_cfg):
    supported_optimizers = {
        'sgd': BuildSGD,
        'adam': BuildAdam,
        'adamw': BuildAdamW,
    }
    assert optimizer_cfg['type'] in supported_optimizers, 'unsupport optimizer type %s' % optimizer_cfg['type']
    selected_optim_cfg = {
        'params_rules': optimizer_cfg.get('params_rules', {}),
        'filter_params': optimizer_cfg.get('filter_params', False)
    }
    selected_optim_cfg.update(optimizer_cfg[optimizer_cfg['type']])
    return supported_optimizers[optimizer_cfg['type']](model, selected_optim_cfg)