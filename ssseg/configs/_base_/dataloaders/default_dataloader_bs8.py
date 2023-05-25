'''default_dataloader_bs8'''
import os


'''DATALOADER_CFG_BS8'''
DATALOADER_CFG_BS8 = {
    'expected_total_bs_for_assert': 8,
    'train': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'test': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
}