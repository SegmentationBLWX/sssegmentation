'''base config for maskformer'''
# config for dataset
DATASET_CFG = {
    'type': '',
    'rootdir': '',
    'train': {
        'set': 'train',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
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
LOSSES_CFG = {}
# config for model
MODEL_CFG = {
    'type': 'maskformer',
    'num_classes': -1,
    'benchmark': True,
    'is_multi_gpus': True,
    'align_corners': False,
    'distributed': {'is_on': True, 'backend': 'nccl'},
    'norm_cfg': {'type': 'syncbatchnorm', 'opts': {}},
    'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    'backbone': {
        'type': 'resnet101',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'ppm': {
        'in_channels': 2048,
        'out_channels': 512,
        'pool_scales': [1, 2, 3, 6],
    },
    'lateral': {
        'in_channels_list': [256, 512, 1024],
        'out_channels': 512,
    },
    'fpn': {
        'in_channels_list': [512, 512, 512],
        'out_channels': 512,
    },
    'decoder': {
        'mask': {'in_channels': 512, 'out_channels': 256},
        'predictor': {
            'in_channels': 2048,
            'mask_classification': True,
            'hidden_dim': 256,
            'num_queries': 100,
            'nheads': 8,
            'dropout': 0.1,
            'dim_feedforward': 2048,
            'enc_layers': 0,
            'dec_layers': 6,
            'pre_norm': False,
            'deep_supervision': True,
            'mask_dim': 256,
            'enforce_input_project': False,
            'norm_cfg': {'type': 'layernorm', 'opts': {}},
            'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
        },
        'matcher': {'cost_class': 1.0, 'cost_mask': 20.0, 'cost_dice': 1.0},
    },
    'auxiliary': None
}
# config for inference
INFERENCE_CFG = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': False
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