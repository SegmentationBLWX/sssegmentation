'''cihp_473x473'''
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_CIHP_473x473'''
DATASET_CFG_CIHP_473x473 = DatasetConfig(
    type='CIHPDataset',
    rootdir=os.path.join(os.getcwd(), 'CIHP'),
    train={
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': (0.75, 1.25)}),
            ('RandomCrop', {'crop_size': (473, 473), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5, 'fixed_seg_target_pairs': [(15, 14), (17, 16), (19, 18)]}),
            ('RandomRotation', {'angle_upper': 30, 'prob': 0.6}),
            ('PhotoMetricDistortion', {}),
            ('EdgeExtractor', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (473, 473), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (473, 473), 'keep_ratio': False, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)