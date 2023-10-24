'''
Function:
    Implementation of BuildPixelSampler
Author:
    Zhenchao Jin
'''
import copy
from .ohempixelsampler import OHEMPixelSampler


'''PixelSamplerBuilder'''
class PixelSamplerBuilder():
    REGISTERED_PIXELSAMPLERS = {
        'OHEMPixelSampler': OHEMPixelSampler,
    }
    def __init__(self, require_register_pixelsamplers=None, require_update_pixelsamplers=None):
        if require_register_pixelsamplers and isinstance(require_register_pixelsamplers, dict):
            for pixelsampler_type, pixelsampler in require_register_pixelsamplers.items():
                self.register(pixelsampler_type, pixelsampler)
        if require_update_pixelsamplers and isinstance(require_update_pixelsamplers, dict):
            for pixelsampler_type, pixelsampler in require_update_pixelsamplers.items():
                self.update(pixelsampler_type, pixelsampler)
    '''build'''
    def build(self, pixelsampler_cfg):
        pixelsampler_cfg = copy.deepcopy(pixelsampler_cfg)
        pixelsampler_type = pixelsampler_cfg.pop('type')
        pixelsampler = self.REGISTERED_PIXELSAMPLERS[pixelsampler_type](**pixelsampler_cfg)
        return pixelsampler
    '''register'''
    def register(self, pixelsampler_type, pixelsampler):
        assert pixelsampler_type not in self.REGISTERED_PIXELSAMPLERS
        self.REGISTERED_PIXELSAMPLERS[pixelsampler_type] = pixelsampler
    '''update'''
    def update(self, pixelsampler_type, pixelsampler):
        assert pixelsampler_type in self.REGISTERED_PIXELSAMPLERS
        self.REGISTERED_PIXELSAMPLERS[pixelsampler_type] = pixelsampler


'''BuildPixelSampler'''
BuildPixelSampler = PixelSamplerBuilder().build