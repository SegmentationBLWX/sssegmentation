'''
Function:
    Base pixel sampler
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''BasePixelSampler'''
class BasePixelSampler(nn.Module):
    def __init__(self):
        pass
    '''placeholder for sample function'''
    def sample(self, seg_logit, seg_label):
        raise NotImplementedError('not to be implemented')