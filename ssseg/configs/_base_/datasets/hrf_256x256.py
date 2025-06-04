'''hrf_256x256'''
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_HRF_256x256'''
DATASET_CFG_HRF_256x256 = DatasetConfig(
    type='HRFDataset',
    rootdir=os.path.join(os.getcwd(), 'HRF'),
    train={
        'set': 'train',
        'repeat_times': 45000,
        'data_pipelines': [
            ('Resize', {'output_size': (2336, 3504), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (256, 256), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (256, 256), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2336, 3504), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)