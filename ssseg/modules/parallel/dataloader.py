'''
Function:
    Implementation of BuildDistributedDataloader
Author:
    Zhenchao Jin
'''
import copy
import torch


'''BuildDistributedDataloader'''
def BuildDistributedDataloader(dataset, dataloader_cfg):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    # build dataloader
    shuffle = dataloader_cfg.pop('shuffle')
    dataloader_cfg['shuffle'] = False
    dataloader_cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_cfg)
    # return
    return dataloader