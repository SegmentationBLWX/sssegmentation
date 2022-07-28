'''
Funtion:
    Build the scheduler
Author:
    Zhenchao Jin
'''
import copy
from .polyscheduler import PolyScheduler


'''BuildScheduler'''
def BuildScheduler(optimizer, scheduler_cfg):
    # define the supported schedulers
    supported_schedulers = {
        'poly': PolyScheduler
    }
    # parse scheduler_cfg
    scheduler_cfg = copy.deepcopy(scheduler_cfg)
    scheduler_type = scheduler_cfg.pop('type')
    scheduler_cfg['optimizer'] = optimizer
    # return
    return supported_schedulers[scheduler_type](**scheduler_cfg)