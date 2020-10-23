'''
Function:
    some utils for optimizers
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad


'''adjust learning rate'''
def adjustLearningRate(optimizer, optimizer_cfg=None):
    # parse and check the config for optimizer
    policy, policy_cfg, type_cfg = optimizer_cfg['policy'], optimizer_cfg['policy_cfg'], optimizer_cfg[optimizer_cfg['type']]
    if ('params_rules' in type_cfg) and (type_cfg['params_rules']):
        assert len(optimizer.param_groups) == len(type_cfg['params_rules'])
    # adjust the learning rate according the policy
    if policy == 'poly':
        base_lr, num_iters, max_iters, power, min_lr = type_cfg['learning_rate'], policy_cfg['num_iters'], policy_cfg['max_iters'], policy_cfg['power'], type_cfg.get('min_lr', 1e-4)
        coeff = (1 - num_iters / max_iters) ** power
        target_lr = coeff * (base_lr - min_lr) + min_lr
        for param_group in optimizer.param_groups:
            if ('params_rules' in type_cfg) and (type_cfg['params_rules']):
                param_group['lr'] = target_lr * type_cfg['params_rules'][param_group['name']]
            else:
                param_group['lr'] = target_lr
    else:
        raise ValueError('Unsupport policy %s...' % policy)
    return target_lr


'''clip gradient'''
def clipGradients(params, max_norm=35, norm_type=2):
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)