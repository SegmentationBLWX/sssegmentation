'''
Funtion:
    Build the optimizer
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.optim as optim


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
    if not params_rules:
        params = model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters())
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers'
        for key, value in params_rules.items():
            if not isinstance(value, tuple): value = (value, value)
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': optimizer_cfg['learning_rate'] * value[0], 
                'name': key,
                'weight_decay': optimizer_cfg['weight_decay'] * value[1],
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        value = (params_rules['others'], params_rules['others']) if not isinstance(params_rules['others'], tuple) else params_rules['others']
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': optimizer_cfg['learning_rate'] * value[0], 
            'name': 'others',
            'weight_decay': optimizer_cfg['weight_decay'] * value[1],
        })
    optimizer_cfg['params'] = params
    # return
    return supported_optimizers[optimizer_type](**optimizer_cfg)