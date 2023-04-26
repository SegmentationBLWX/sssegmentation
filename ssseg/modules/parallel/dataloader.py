'''
Function:
    Implementation of BuildDistributedDataloader
Author:
    Zhenchao Jin
'''
import torch


'''BuildDistributedDataloader'''
def BuildDistributedDataloader(dataset, dataloader_cfg):
    args = {
        'batch_size': dataloader_cfg.get('batch_size', 16),
        'num_workers': dataloader_cfg.get('num_workers', 16),
        'shuffle': False,
        'pin_memory': dataloader_cfg.get('pin_memory', True),
        'drop_last': dataloader_cfg.get('drop_last', True),
        'sampler': torch.utils.data.distributed.DistributedSampler(dataset, shuffle=dataloader_cfg.get('shuffle', True)),
    }
    dataloader = torch.utils.data.DataLoader(dataset, **args)
    return dataloader