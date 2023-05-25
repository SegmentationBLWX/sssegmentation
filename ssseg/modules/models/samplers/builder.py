'''
Function:
    Implementation of BuildPixelSampler
Author:
    Zhenchao Jin
'''
import copy
from .ohempixelsampler import OHEMPixelSampler


'''BuildPixelSampler'''
def BuildPixelSampler(sampler_cfg):
    sampler_cfg = copy.deepcopy(sampler_cfg)
    # supported samplers
    supported_samplers = {
        'OHEMPixelSampler': OHEMPixelSampler,
    }
    # build sampler
    sampler_type = sampler_cfg.pop('type')
    sampler = supported_samplers[sampler_type](**sampler_cfg)
    # return
    return sampler