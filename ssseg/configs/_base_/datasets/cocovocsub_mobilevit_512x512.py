'''cocovocsub_mobilevit_512x512'''
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_COCOVOCSUB_MOBILEVIT_512x512'''
DATASET_CFG_COCOVOCSUB_MOBILEVIT_512x512 = DatasetConfig(
    type='COCOVOCSUBDataset',
    rootdir=os.path.join(os.getcwd(), 'COCO'),
    train={
        'set': 'train',
        'data_pipelines': [
            ('RandomShortestEdgeResize', {'short_edge_range': (256, 768), 'max_size': 1024}),
            ('RandomFlip', {'prob': 0.5}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('PILRandomGaussianBlur', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('RandomRotation', {'angle_upper': 10, 'prob': 1.0}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)