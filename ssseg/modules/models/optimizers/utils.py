'''
Function:
    Some utils for optimizers
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad


'''adjustLearningRate'''
def adjustLearningRate(optimizer, optimizer_cfg=None):
    # get warmup lr
    def getwarmuplr(cur_iters, warmup_cfg, regular_lr):
        warmup_type, warmup_ratio, warmup_iters = warmup_cfg['type'], warmup_cfg['ratio'], warmup_cfg['iters']
        if warmup_type == 'constant':
            warmup_lr = regular_lr * warmup_ratio
        elif warmup_type == 'linear':
            k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
            warmup_lr = (1 - k) * regular_lr
        elif warmup_type == 'exp':
            k = warmup_ratio**(1 - cur_iters / warmup_iters)
            warmup_lr = k * regular_lr
        return warmup_lr
    # parse and check the config for optimizer
    policy_cfg, selected_optim_cfg, warmup_cfg = optimizer_cfg['policy'], optimizer_cfg[optimizer_cfg['type']], None
    if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
        assert len(optimizer.param_groups) == len(optimizer_cfg['params_rules'])
    if ('warmup' in policy_cfg) and (policy_cfg['warmup']):
        warmup_cfg = policy_cfg['warmup']
    # adjust the learning rate according the policy
    if policy_cfg['type'] == 'poly':
        base_lr = selected_optim_cfg['learning_rate']
        min_lr = selected_optim_cfg.get('min_lr', base_lr * 0.01)
        num_iters, max_iters, power = policy_cfg['opts']['num_iters'], policy_cfg['opts']['max_iters'], policy_cfg['opts']['power']
        coeff = (1 - num_iters / max_iters) ** power
        target_lr = coeff * (base_lr - min_lr) + min_lr
        if (warmup_cfg is not None) and (warmup_cfg['iters'] >= num_iters):
            target_lr = getwarmuplr(num_iters, warmup_cfg, target_lr)
        for param_group in optimizer.param_groups:
            if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
                value = optimizer_cfg['params_rules'][param_group['name']]
                if not isinstance(value, tuple): value = (value, value)
                param_group['lr'] = target_lr * value[0]
            else:
                param_group['lr'] = target_lr
    else:
        raise ValueError('Unsupport policy %s' % policy)
    return target_lr


'''clipGradients'''
def clipGradients(params, max_norm=35, norm_type=2):
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)