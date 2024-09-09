'''
Function:
    Implementation of BuildDistributedModel
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn


'''BuildDistributedModel'''
def BuildDistributedModel(model, model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    return nn.parallel.DistributedDataParallel(model, **model_cfg)