'''
Funtion:
    Implementation of SchedulerBuilder and BuildScheduler
Author:
    Zhenchao Jin
'''
import copy
from .polyscheduler import PolyScheduler


'''SchedulerBuilder'''
class SchedulerBuilder():
    REGISTERED_SCHEDULERS = {
        'PolyScheduler': PolyScheduler,
    }
    def __init__(self, require_register_schedulers=None, require_update_schedulers=None):
        if require_register_schedulers and isinstance(require_register_schedulers, dict):
            for scheduler_type, scheduler in require_register_schedulers.items():
                self.register(scheduler_type, scheduler)
        if require_update_schedulers and isinstance(require_update_schedulers, dict):
            for scheduler_type, scheduler in require_update_schedulers.items():
                self.update(scheduler_type, scheduler)
    '''build'''
    def build(self, optimizer, scheduler_cfg):
        scheduler_cfg = copy.deepcopy(scheduler_cfg)
        scheduler_type = scheduler_cfg.pop('type')
        scheduler_cfg.pop('optimizer')
        scheduler = self.REGISTERED_SCHEDULERS[scheduler_type](optimizer=optimizer, **scheduler_cfg)
        return scheduler
    '''register'''
    def register(self, scheduler_type, scheduler):
        assert scheduler_type not in self.REGISTERED_SCHEDULERS
        self.REGISTERED_SCHEDULERS[scheduler_type] = scheduler
    '''update'''
    def update(self, scheduler_type, scheduler):
        assert scheduler_type in self.REGISTERED_SCHEDULERS
        self.REGISTERED_SCHEDULERS[scheduler_type] = scheduler


'''BuildScheduler'''
BuildScheduler = SchedulerBuilder().build