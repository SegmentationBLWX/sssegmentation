'''
Function:
    Implementation of some utils
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn


'''getclones'''
def getclones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])