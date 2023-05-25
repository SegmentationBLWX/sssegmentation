'''atr_473x473'''
import os


'''DATASET_CFG_ATR_473x473'''
DATASET_CFG_ATR_473x473 = {
    'type': 'ATRDataset',
    'rootdir': os.path.join(os.getcwd(), 'ATR'),
    'train': {
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': (0.75, 1.25)}),
            ('RandomCrop', {'crop_size': (473, 473), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5, 'fixed_seg_target_pairs': [(9, 10), (12, 13), (14, 15)]}),
            ('RandomRotation', {'angle_upper': 30, 'rotation_prob': 0.6}),
            ('PhotoMetricDistortion', {}),
            ('EdgeExtractor', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (473, 473), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (473, 473), 'keep_ratio': False, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}