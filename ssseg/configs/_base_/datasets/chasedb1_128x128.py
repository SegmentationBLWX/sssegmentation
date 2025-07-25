'''chasedb1_128x128'''
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_CHASEDB1_128x128'''
DATASET_CFG_CHASEDB1_128x128 = DatasetConfig(
    type='ChaseDB1Dataset',
    rootdir=os.path.join(os.getcwd(), 'CHASE_DB1'),
    train={
        'set': 'train',
        'repeat_times': 35000,
        'data_pipelines': [
            ('Resize', {'output_size': (960, 999), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (128, 128), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (128, 128), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (960, 999), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)