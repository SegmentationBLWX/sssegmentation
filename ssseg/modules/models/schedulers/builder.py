'''
Function:
    Implementation of SchedulerBuilder and BuildScheduler
Author:
    Zhenchao Jin
'''
import copy
from ...utils import BaseModuleBuilder
from .polyscheduler import PolyScheduler
from .cosinescheduler import CosineScheduler


'''SchedulerBuilder'''
class SchedulerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'PolyScheduler': PolyScheduler, 'CosineScheduler': CosineScheduler,
    }
    '''build'''
    def build(self, optimizer, scheduler_cfg):
        scheduler_cfg = copy.deepcopy(scheduler_cfg)
        scheduler_type = scheduler_cfg.pop('type')
        scheduler_cfg.pop('optimizer')
        scheduler = self.REGISTERED_MODULES[scheduler_type](optimizer=optimizer, **scheduler_cfg)
        return scheduler


'''BuildScheduler'''
BuildScheduler = SchedulerBuilder().build