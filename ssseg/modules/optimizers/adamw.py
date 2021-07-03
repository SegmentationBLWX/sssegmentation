'''
Funtion:
    define AdamW optimizer
Author:
    Zhenchao Jin
'''
import torch.nn as nn
import torch.optim as optim


'''AdamW builder'''
def BuildAdamW(model, cfg, **kwargs):
    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    if not params_rules:
        params = model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters())
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if not isinstance(value, tuple): value = (value, value)
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value[0], 
                'name': key,
                'weight_decay': cfg['weight_decay'] * value[1],
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        value = (params_rules['others'], params_rules['others']) if not isinstance(params_rules['others'], tuple) else params_rules['others']
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * value[0], 
            'name': 'others',
            'weight_decay': cfg['weight_decay'] * value[1],
        })
    optimizer = optim.AdamW(
        params,
        lr=cfg['learning_rate'],
        betas=cfg.get('betas', (0.9, 0.999)),
        eps=cfg.get('eps', 1e-08),
        weight_decay=cfg['weight_decay'],
        amsgrad=cfg.get('amsgrad', False)
    )
    return optimizer