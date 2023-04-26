'''
Function:
    Implementation of HardSwish
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''HardSwish'''
class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    '''forward'''
    def forward(self, x):
        return x * self.act(x + 3) / 6