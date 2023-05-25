'''
Funtion:
    Implementation of BuildScheduler
Author:
    Zhenchao Jin
'''
import copy
from .polyscheduler import PolyScheduler


'''BuildScheduler'''
def BuildScheduler(optimizer, scheduler_cfg):
    scheduler_cfg = copy.deepcopy(scheduler_cfg)
    # supported schedulers
    supported_schedulers = {
        'PolyScheduler': PolyScheduler
    }
    # build scheduler
    scheduler_type = scheduler_cfg.pop('type')
    scheduler_cfg.pop('optimizer')
    scheduler = supported_schedulers[scheduler_type](optimizer=optimizer, **scheduler_cfg)
    # return
    return scheduler