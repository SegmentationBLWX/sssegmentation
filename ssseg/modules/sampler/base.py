'''
Function:
    base pixel sampler
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''base pixel sampler'''
class BasePixelSampler(nn.Module):
    def __init__(self, **kwargs):
        pass
    '''placeholder for sample function'''
    def sample(self, seg_logit, seg_label):
        raise NotImplementedError('not to be implemented')