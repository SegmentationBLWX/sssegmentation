'''
Funtion:
    Define the basescheduler
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad


'''BaseScheduler'''
class BaseScheduler():
    def __init__(self, optimizer=None, lr=0.01, min_lr=None, warmup_cfg=None, max_epochs=-1, iters_per_epoch=-1, params_rules=dict()):
        # set attrs
        self.lr = lr
        self.min_lr = min_lr if min_lr is not None else lr * 0.01
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_cfg = warmup_cfg
        self.params_rules = params_rules
        self.max_iters = max_epochs * iters_per_epoch
        # initialize some variables
        self.cur_epoch = 0
        self.cur_iter = 0
    '''step'''
    def step(self):
        raise NotImplementedError('not to be implemented')
    '''update lr'''
    def updatelr(self):
        raise NotImplementedError('not to be implemented')
    '''get warmup lr'''
    def getwarmuplr(self, cur_iter, warmup_cfg, regular_lr):
        warmup_type, warmup_ratio, warmup_iters = warmup_cfg['type'], warmup_cfg['ratio'], warmup_cfg['iters']
        if warmup_type == 'constant':
            warmup_lr = regular_lr * warmup_ratio
        elif warmup_type == 'linear':
            k = (1 - cur_iter / warmup_iters) * (1 - warmup_ratio)
            warmup_lr = (1 - k) * regular_lr
        elif warmup_type == 'exp':
            k = warmup_ratio**(1 - cur_iter / warmup_iters)
            warmup_lr = k * regular_lr
        return warmup_lr
    '''clip gradients'''
    def clipgradients(self, params, max_norm=35, norm_type=2):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)
    '''state'''
    def state(self):
        state_dict = {
            'cur_epoch': self.cur_epoch, 'cur_iter': self.cur_iter,
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict
    '''setstate'''
    def setstate(self, state_dict):
        self.cur_epoch = state_dict['cur_epoch']
        self.cur_iter = state_dict['cur_iter']