'''
Function:
    Implementation of CosineScheduler
Author:
    Zhenchao Jin
'''
import math
from .basescheduler import BaseScheduler


'''CosineScheduler'''
class CosineScheduler(BaseScheduler):
    def __init__(self, by_epoch=True, optimizer=None, lr=0.01, min_lr=None, warmup_cfg=None, clipgrad_cfg=None, max_epochs=-1, iters_per_epoch=-1, params_rules=dict()):
        super(CosineScheduler, self).__init__(
            optimizer=optimizer, lr=lr, min_lr=min_lr, warmup_cfg=warmup_cfg, clipgrad_cfg=clipgrad_cfg, 
            max_epochs=max_epochs, iters_per_epoch=iters_per_epoch, params_rules=params_rules,
        )
        self.by_epoch = by_epoch
    '''updatelr'''
    def updatelr(self):
        # obtain variables
        base_lr, min_lr, cur_iter, cur_epoch, max_iters, max_epochs, by_epoch = self.lr, self.min_lr, self.cur_iter, self.cur_epoch, self.max_iters, self.max_epochs, self.by_epoch
        optimizer, warmup_cfg, params_rules = self.optimizer, self.warmup_cfg, self.params_rules
        # calculate target learning rate
        if by_epoch:
            target_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * cur_epoch / max_epochs))
        else:
            target_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * cur_iter / max_iters))
        if (warmup_cfg is not None) and (warmup_cfg['iters'] >= cur_iter):
            target_lr = self.getwarmuplr(cur_iter, warmup_cfg, target_lr)
        # update learning rate
        for param_group in optimizer.param_groups:
            if params_rules and params_rules.get('type', 'DefaultParamsConstructor') == 'DefaultParamsConstructor':
                param_group['lr'] = param_group.get('lr_multiplier', 1.0) * target_lr
            elif params_rules and params_rules.get('type', 'DefaultParamsConstructor') == 'LearningRateDecayParamsConstructor':
                param_group['lr'] = param_group['lr_multiplier'] * target_lr
            else:
                param_group['lr'] = target_lr
        # return
        return target_lr