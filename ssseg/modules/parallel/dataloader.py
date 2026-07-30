'''
Function:
    Implementation of BuildDistributedDataloader
Author:
    Zhenchao Jin
'''
import copy
import torch
import random
import numpy as np
import torch.utils.data.distributed


'''seedworker'''
def seedworker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


'''BuildDistributedDataloader'''
def BuildDistributedDataloader(dataset, dataloader_cfg: dict):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    # build dataloader
    shuffle = dataloader_cfg.pop('shuffle')
    dataloader_cfg['shuffle'] = False
    dataloader_cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader_cfg.setdefault("worker_init_fn", seedworker,)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_cfg)
    # return
    return dataloader