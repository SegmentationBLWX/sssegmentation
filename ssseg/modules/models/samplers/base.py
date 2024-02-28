'''
Function:
    Implementation of BasePixelSampler
Author:
    Zhenchao Jin
'''
import torch.nn as nn


'''BasePixelSampler'''
class BasePixelSampler(nn.Module):
    def __init__(self):
        pass
    '''sample'''
    def sample(self, seg_logits, seg_targets):
        raise NotImplementedError('not to be implemented')