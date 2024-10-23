'''drive_64x64'''
import os


'''DATASET_CFG_DRIVE_64x64'''
DATASET_CFG_DRIVE_64x64 = {
    'type': 'DRIVEDataset',
    'rootdir': os.path.join(os.getcwd(), 'DRIVE'),
    'train': {
        'set': 'train',
        'repeat_times': 32000,
        'data_pipelines': [
            ('Resize', {'output_size': (584, 565), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (64, 64), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (64, 64), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (584, 565), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}