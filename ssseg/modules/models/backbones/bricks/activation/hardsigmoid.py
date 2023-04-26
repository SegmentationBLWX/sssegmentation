'''
Function:
    Implementation of HardSigmoid
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''HardSigmoid'''
class HardSigmoid(nn.Module):
    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value
    '''forward'''
    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)