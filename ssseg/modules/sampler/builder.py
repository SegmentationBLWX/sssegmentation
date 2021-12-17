'''
Function:
    Build the pixel sampler
Author:
    Zhenchao Jin
'''
from .ohempixelsampler import OHEMPixelSampler


'''build pixel sampler'''
def BuildPixelSampler(sampler_type='ohem', sampler_cfg=None, **kwargs):
    supported_samplers = {
        'ohem': OHEMPixelSampler,
    }
    assert sampler_type in supported_samplers, 'unsupport loss type %s...' % loss_type
    return supported_samplers[sampler_type](**sampler_cfg)