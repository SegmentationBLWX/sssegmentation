'''
Function:
    Build the pixel sampler
Author:
    Zhenchao Jin
'''
import copy
from .ohempixelsampler import OHEMPixelSampler


'''BuildPixelSampler'''
def BuildPixelSampler(sampler_cfg):
    supported_samplers = {
        'ohem': OHEMPixelSampler,
    }
    selected_sampler = supported_samplers[sampler_cfg['type']]
    sampler_cfg = copy.deepcopy(sampler_cfg)
    sampler_cfg.pop('type')
    return selected_sampler(**sampler_cfg)