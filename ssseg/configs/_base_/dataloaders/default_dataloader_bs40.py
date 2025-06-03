'''default_dataloader_bs40'''
from .default_dataloader import DataloaderConfig


'''DATALOADER_CFG_BS40'''
DATALOADER_CFG_BS40 = DataloaderConfig(
    expected_total_train_bs_for_assert=40,
    auto_adapt_to_expected_train_bs=True,
    train={
        'batch_size_per_gpu': 5, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True
    },
    test={
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False
    },
)