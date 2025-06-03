'''pascalcontext_480x480'''
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_PASCALCONTEXT_480x480'''
DATASET_CFG_PASCALCONTEXT_480x480 = DatasetConfig(
    type='PascalContextDataset',
    rootdir=os.path.join(os.getcwd(), 'VOCdevkit/VOC2010/'),
    train={
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (520, 520), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (480, 480), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (480, 480), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (520, 520), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)