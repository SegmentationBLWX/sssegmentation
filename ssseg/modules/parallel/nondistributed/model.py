'''
Function:
    non-distributed model
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''non-distributed model'''
def NonDistributedModel(model, **kwargs):
    return nn.parallel.DataParallel(model, **kwargs)