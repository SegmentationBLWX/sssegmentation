'''
Function:
    non-distributed dataloader
Author:
    Zhenchao Jin
'''
import torch


'''non-distributed dataloader'''
def NonDistributedDataloader(dataset, cfg, **kwargs):
    args = {
        'batch_size': cfg.get('batch_size', 16),
        'num_workers': cfg.get('num_workers', 16),
        'shuffle': cfg.get('shuffle', True),
        'pin_memory': cfg.get('pin_memory', True),
        'drop_last': cfg.get('drop_last', True)
    }
    dataloader = torch.utils.data.DataLoader(dataset, **args)
    return dataloader