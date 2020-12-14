'''
Funtion:
    define SGD optimizer
Author:
    Zhenchao Jin
'''
import torch.nn as nn
import torch.optim as optim


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