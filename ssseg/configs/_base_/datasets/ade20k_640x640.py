'''ade20k_640x640'''
import os


'''DATASET_CFG_ADE20k_640x640'''
DATASET_CFG_ADE20k_640x640 = {
    'type': 'ADE20kDataset',
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'),
    'train': {
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (2560, 640), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (640, 640), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (640, 640), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2560, 640), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}