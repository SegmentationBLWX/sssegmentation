'''
Function:
    Build distributed model
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''BuildDistributedModel'''
def BuildDistributedModel(model, model_cfg):
    return nn.parallel.DistributedDataParallel(model, **model_cfg)