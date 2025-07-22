'''cityscapes_768x768'''
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_CITYSCAPES_768x768'''
DATASET_CFG_CITYSCAPES_768x768 = DatasetConfig(
    type='CityScapesDataset',
    rootdir=os.path.join(os.getcwd(), 'CityScapes'),
    train={
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (2049, 1025), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (768, 768), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (768, 768), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2049, 1025), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)