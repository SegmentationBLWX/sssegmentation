'''
Function:
    distributed model
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''distributed model'''
def DistributedModel(model, **kwargs):
    return nn.parallel.DistributedDataParallel(model, **kwargs)