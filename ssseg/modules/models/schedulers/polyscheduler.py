'''
Funtion:
    Define the polyscheduler
Author:
    Zhenchao Jin
'''
from .basescheduler import BaseScheduler


'''PolyScheduler'''
class PolyScheduler(BaseScheduler):
    def __init__(self, power=0.9, optimizer=None, lr=0.01, min_lr=None, warmup_cfg=None, max_epochs=-1, iters_per_epoch=-1, params_rules=dict()):
        super(PolyScheduler, self).__init__(
            optimizer=optimizer, lr=lr, min_lr=min_lr, warmup_cfg=warmup_cfg, max_epochs=max_epochs, iters_per_epoch=iters_per_epoch, params_rules=params_rules,
        )
        self.power = power
    '''update lr'''
    def updatelr(self):
        # obtain variables
        base_lr, min_lr, cur_iter, max_iters, power = self.lr, self.min_lr, self.cur_iter, self.max_iters, self.power
        optimizer, warmup_cfg, params_rules = self.optimizer, self.warmup_cfg, self.params_rules
        if params_rules and params_rules.get('type', 'default') == 'default':
            assert len(optimizer.param_groups) == len(params_rules)
        # calculate target learning rate
        coeff = (1 - cur_iter / max_iters) ** power
        target_lr = coeff * (base_lr - min_lr) + min_lr
        if (warmup_cfg is not None) and (warmup_cfg['iters'] >= cur_iter):
            target_lr = self.getwarmuplr(cur_iter, warmup_cfg, target_lr)
        # update learning rate
        for param_group in optimizer.param_groups:
            if params_rules and params_rules.get('type', 'default') == 'default':
                value = params_rules[param_group['name']]
                if not isinstance(value, tuple):
                    value = (value, value)
                param_group['lr'] = target_lr * value[0]
            elif params_rules and params_rules.get('type', 'default') == 'layerdecay':
                param_group['lr'] = param_group['lr_scale'] * target_lr
            else:
                param_group['lr'] = target_lr
        # return
        return target_lr
    '''step'''
    def step(self):
        self.optimizer.step()
        self.cur_iter += 1