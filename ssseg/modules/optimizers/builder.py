'''
Funtion:
    define the optimizer builders
Author:
    Zhenchao Jin
'''
import torch.nn as nn
import torch.optim as optim


'''define all'''
__all__ = ['BuildOptimizer']


'''sgd builder'''
def SGDBuilder(model, cfg, **kwargs):
    params_rules = cfg.get('params_rules', {})
    if not params_rules:
        optimizer = optim.SGD(model.parameters(), 
                              lr=cfg['learning_rate'], 
                              momentum=cfg['momentum'], 
                              weight_decay=cfg['weight_decay'],
                              **kwargs)
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({'params': all_layers[key].parameters(), 'lr': cfg['learning_rate'] * value, 'name': key})
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        params.append({'params': others.parameters(), 'lr': cfg['learning_rate'] * params_rules['others'], 'name': 'others'})
        optimizer = optim.SGD(params, 
                              lr=cfg['learning_rate'], 
                              momentum=cfg['momentum'], 
                              weight_decay=cfg['weight_decay'],
                              **kwargs)
    return optimizer


'''adam builder'''
def AdamBuilder(model, cfg, **kwargs):
    params_rules = cfg.get('params_rules', {})
    if not params_rules:
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               **kwargs)
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({'params': all_layers[key].parameters(), 'lr': cfg['learning_rate'] * value, 'name': key})
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        params.append({'params': others.parameters(), 'lr': cfg['learning_rate'] * params_rules['others'], 'name': 'others'})
        optimizer = optim.Adam(params,
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               **kwargs)
    return optimizer


'''build optimizer'''
def BuildOptimizer(model, cfg, **kwargs):
    supported_dict = {
        'sgd': SGDBuilder,
        'adam': AdamBuilder
    }
    assert cfg['type'] in supported_dict, 'unsupport optimizer type %s...' % cfg['type']
    return supported_dict[cfg['type']](model, cfg[cfg['type']])