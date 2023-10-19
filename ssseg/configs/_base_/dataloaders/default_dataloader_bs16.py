'''default_dataloader_bs16'''
import os


'''DATALOADER_CFG_BS16'''
DATALOADER_CFG_BS16 = {
    'expected_total_train_bs_for_assert': 16,
    'auto_adapt_to_expected_train_bs': True,
    'train': {
        'batch_size_per_gpu': 2, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'test': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
}