'''
Funtion:
    define the optimizer builder
Author:
    Zhenchao Jin
'''
from .sgd import SGDBuilder
from .adam import AdamBuilder


'''build optimizer'''
def BuildOptimizer(model, cfg, **kwargs):
    supported_optimizers = {
        'sgd': SGDBuilder,
        'adam': AdamBuilder
    }
    assert cfg['type'] in supported_optimizers, 'unsupport optimizer type %s...' % cfg['type']
    selected_optim_cfg = {
        'params_rules': cfg.get('params_rules', {}),
        'filter_params': cfg.get('filter_params', False)
    }
    selected_optim_cfg.update(cfg[cfg['type']])
    return supported_optimizers[cfg['type']](model, selected_optim_cfg, **kwargs)