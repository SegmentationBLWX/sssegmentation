'''default_dataloader_bs64'''
import os


'''DATALOADER_CFG_BS64'''
DATALOADER_CFG_BS64 = {
    'expected_total_train_bs_for_assert': 64,
    'auto_adapt_to_expected_train_bs': True,
    'train': {
        'batch_size_per_gpu': 8, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'test': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
}