'''
Funtion:
    Build Adam optimizer
Author:
    Zhenchao Jin
'''
import torch.nn as nn
import torch.optim as optim


'''BuildAdam'''
def BuildAdam(model, build_cfg):
    params_rules, filter_params = build_cfg.get('params_rules', {}), build_cfg.get('filter_params', False)
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
                'lr': build_cfg['learning_rate'] * value[0], 
                'name': key,
                'weight_decay': build_cfg['weight_decay'] * value[1],
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        value = (params_rules['others'], params_rules['others']) if not isinstance(params_rules['others'], tuple) else params_rules['others']
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': build_cfg['learning_rate'] * value[0], 
            'name': 'others',
            'weight_decay': build_cfg['weight_decay'] * value[1],
        })
    optimizer = optim.Adam(
        params,
        lr=build_cfg['learning_rate'],
        weight_decay=build_cfg['weight_decay'],
        betas=build_cfg.get('betas', (0.9, 0.999)),
        eps=build_cfg.get('eps', 1e-08),
        amsgrad=build_cfg.get('amsgrad', False)
    )
    return optimizer