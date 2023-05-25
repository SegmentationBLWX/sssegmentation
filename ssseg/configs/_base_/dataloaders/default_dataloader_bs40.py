'''default_dataloader_bs40'''
import os


'''DATALOADER_CFG_BS40'''
DATALOADER_CFG_BS40 = {
    'expected_total_train_bs_for_assert': 40,
    'train': {
        'batch_size_per_gpu': 5, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'test': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
}