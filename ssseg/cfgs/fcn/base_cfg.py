'''base config for fcn'''
# config for dataset
DATASET_CFG = {
    'train': {
        'type': '',
        'set': 'train',
        'rootdir': '',
        'aug_opts': [('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
                     ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
                     ('RandomFlip', {'flip_prob': 0.5}),
                     ('PhotoMetricDistortion', {}),
                     ('RandomRotation', {'angle_upper': 10, 'rotation_prob': 0.6}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),
                     ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),]
    },
    'test': {
        'type': '',
        'set': 'val',
        'rootdir': '',
        'aug_opts': [('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),]
    }
}
# config for dataloader
DATALOADER_CFG = {
    'train': {
                'type': ['nondistributed', 'distributed'][1],
                'batch_size': 16,
                'num_workers': 16,
                'shuffle': True,
                'pin_memory': True,
                'drop_last': True,
        },
    'test': {
                'type': ['nondistributed', 'distributed'][1],
                'batch_size': 1,
                'num_workers': 16,
                'shuffle': False,
                'pin_memory': True,
                'drop_last': False,
        }
}
# config for optimizer
OPTIMIZER_CFG = {
    'type': 'sgd',
    'sgd': {
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'max_epochs': 0,
    'params_rules': {},
    'policy': {
        'type': 'poly',
        'opts': {'power': 0.9, 'max_iters': None, 'num_iters': None, 'num_epochs': None}
    },
    'adjust_period': ['iteration', 'epoch'][0],
}
# config for losses
LOSSES_CFG = {
                'auxiliary': {
                                'celoss': {'scale_factor': 0.4, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
                            },
                'classification': {
                                    'celoss': {'scale_factor': 1.0, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
                                },   
}
# config for model
MODEL_CFG = {
    'distributed': {'is_on': True, 'backend': 'nccl'},
    'benchmark': True,
    'type': 'fcn',
    'num_classes': -1,
    'is_multi_gpus': True,
    'normlayer_opts': {'type': 'syncbatchnorm2d', 'opts': {}},
    'backbone': {
                'type': 'resnet101',
                'series': 'resnet',
                'pretrained': True,
                'outstride': 8,
                'is_improved_version': True
        },
    'decoder': {
            'in_channels': 2048,
            'out_channels': 512,
            'dropout': 0.1,
        },
    'auxiliary': {
            'in_channels': 1024,
            'out_channels': 512,
            'dropout': 0.1,
        }
}
# config for common
COMMON_CFG = {
    'train': {
            'backupdir': '',
            'logfilepath': '',
            'loginterval': 50,
            'saveinterval': 1
        },
    'test': {
            'backupdir': '',
            'logfilepath': '',
            'resultsavepath': ''
        }
}