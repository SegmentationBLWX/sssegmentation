'''fcn_cgnetm3n21_cityscapes'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['CGNET_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_CITYSCAPES_512x1024'].copy()
SEGMENTOR_CFG['dataset']['train']['data_pipelines'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [72.39239876, 82.90891754, 73.15835921], 'std': [1, 1, 1]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),
]
SEGMENTOR_CFG['dataset']['test']['data_pipelines'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [72.39239876, 82.90891754, 73.15835921], 'std': [1, 1, 1]}),
    ('ToTensor', {}),
]
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 340
SEGMENTOR_CFG['scheduler']['min_lr'] = 1e-4
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['losses'] = {
    'loss_cls': {
        'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean', 
        'class_weight': [2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352, 10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905, 10.347791, 6.3927646, 10.226669, 10.241062, 10.280587, 10.396974, 10.055647],
    },
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")