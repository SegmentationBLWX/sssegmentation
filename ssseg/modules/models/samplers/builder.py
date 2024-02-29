'''
Function:
    Implementation of PixelSamplerBuilder and BuildPixelSampler
Author:
    Zhenchao Jin
'''
from .ugsampler import UGSampler
from ...utils import BaseModuleBuilder
from .ohempixelsampler import OHEMPixelSampler


'''PixelSamplerBuilder'''
class PixelSamplerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'OHEMPixelSampler': OHEMPixelSampler, 'UGSampler': UGSampler,
    }
    '''build'''
    def build(self, pixelsampler_cfg):
        return super().build(pixelsampler_cfg)


'''BuildPixelSampler'''
BuildPixelSampler = PixelSamplerBuilder().build