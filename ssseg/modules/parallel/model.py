'''
Function:
    Implementation of BuildDistributedModel
Author:
    Zhenchao Jin
'''
import torch.nn as nn


'''BuildDistributedModel'''
def BuildDistributedModel(model, model_cfg):
    return nn.parallel.DistributedDataParallel(model, **model_cfg)