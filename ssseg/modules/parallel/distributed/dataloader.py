'''
Function:
    distributed dataloader
Author:
    Zhenchao Jin
'''
import torch


'''distributed dataloader'''
def DistributedDataloader(dataset, cfg, **kwargs):
    args = {
        'batch_size': cfg.get('batch_size', 16),
        'num_workers': cfg.get('num_workers', 16),
        'shuffle': False,
        'pin_memory': cfg.get('pin_memory', True),
        'drop_last': cfg.get('drop_last', True),
        'sampler': torch.utils.data.distributed.DistributedSampler(dataset, shuffle=cfg.get('shuffle', True)),
    }
    dataloader = torch.utils.data.DataLoader(dataset, **args)
    return dataloader