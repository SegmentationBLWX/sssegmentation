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
    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    if not params_rules:
        optimizer = optim.SGD(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=cfg['learning_rate'], 
                              momentum=cfg['momentum'], 
                              weight_decay=cfg['weight_decay'],
                              dampening=cfg.get('dampening', 0),
                              nesterov=cfg.get('nesterov', False))
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value, 
                'name': key
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * params_rules['others'], 
            'name': 'others'
        })
        optimizer = optim.SGD(params, 
                              lr=cfg['learning_rate'], 
                              momentum=cfg['momentum'], 
                              weight_decay=cfg['weight_decay'],
                              dampening=cfg.get('dampening', 0),
                              nesterov=cfg.get('nesterov', False))
    return optimizer


'''adam builder'''
def AdamBuilder(model, cfg, **kwargs):
    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    if not params_rules:
        optimizer = optim.Adam(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               betas=cfg.get('betas', (0.9, 0.999)),
                               eps=cfg.get('eps', 1e-08),
                               amsgrad=cfg.get('amsgrad', False))
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value, 
                'name': key
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * params_rules['others'], 
            'name': 'others'
        })
        optimizer = optim.Adam(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               betas=cfg.get('betas', (0.9, 0.999)),
                               eps=cfg.get('eps', 1e-08),
                               amsgrad=cfg.get('amsgrad', False))
    return optimizer


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
    return supported_optimizers[cfg['type']](model, selected_optim_cfg)