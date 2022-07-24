'''base config for pointrend'''
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
        'batch_size': 16, 'num_workers': 16, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'test': {
        'batch_size': 1, 'num_workers': 16, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
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
        'opts': {'power': 0.9, 'max_iters': None, 'num_iters': None, 'num_epochs': None},
        'warmup': {'type': 'linear', 'ratio': 0.1, 'iters': 200},
    },
    'adjust_period': ['iteration', 'epoch'][0],
}
# config for losses
LOSSES_CFG = {
    'loss_aux': {
        'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'loss_cls': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
}
# config for segmentor
SEGMENTOR_CFG = {
    'type': 'pointrend',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'backend': 'nccl',
    'norm_cfg': {'type': 'syncbatchnorm'},
    'act_cfg': {'type': 'relu', 'inplace': True},
    'backbone': {
        'type': 'resnet101',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 32,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'fpn': {
        'in_channels_list': [256, 512, 1024, 2048],
        'out_channels': 256,
        'upsample_cfg': {'mode': 'nearest'},
        'feature_stride_list': [4, 8, 16, 32],
        'scale_head_channels': 128,
    },
    'pointrend': {
        'num_fcs': 3,
        'in_channels_list': [256],
        'feats_channels': 256,
        'coarse_pred_each_layer': True,
        'train': {'num_points': 2048, 'oversample_ratio': 3, 'importance_sample_ratio': 0.75},
        'test': {'subdivision_steps': 2, 'subdivision_num_points': 8196, 'scale_factor': 2},
    },
    'decoder': {
        'dropout': 0,
    },
    'auxiliary': {
        'in_channels': 128,
        'dropout': 0,
        'num_convs': 0,
    }
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
    'work_dir': 'ckpts',
    'logfilepath': '',
    'log_interval_iterations': 50,
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'resultsavepath': '',
}