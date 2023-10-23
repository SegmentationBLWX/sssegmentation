'''
Function:
    Implementation of Swish
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''Swish'''
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    '''forward'''
    def forward(self, x):
        return x * torch.sigmoid(x)