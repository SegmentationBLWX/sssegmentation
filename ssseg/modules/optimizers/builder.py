'''
Funtion:
    Build the optimizer
Author:
    Zhenchao Jin
'''
from .sgd import BuildSGD
from .adam import BuildAdam
from .adamw import BuildAdamW


'''build optimizer'''
def BuildOptimizer(model, cfg, **kwargs):
    supported_optimizers = {
        'sgd': BuildSGD,
        'adam': BuildAdam,
        'adamw': BuildAdamW,
    }
    assert cfg['type'] in supported_optimizers, 'unsupport optimizer type %s...' % cfg['type']
    selected_optim_cfg = {
        'params_rules': cfg.get('params_rules', {}),
        'filter_params': cfg.get('filter_params', False)
    }
    selected_optim_cfg.update(cfg[cfg['type']])
    return supported_optimizers[cfg['type']](model, selected_optim_cfg, **kwargs)