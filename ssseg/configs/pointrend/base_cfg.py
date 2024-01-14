'''SEGMENTOR_CFG for PointRend'''
SEGMENTOR_CFG = {
    'type': 'PointRend',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'backend': 'nccl',
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'evaluate_results_filename': '',
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 50},
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
        'pretrained': True, 'outstride': 32, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'fpn_in_channels_list': [256, 512, 1024, 2048], 'pointrend_in_channels_list': [256], 'feats_channels': 256,
        'upsample_cfg': {'mode': 'nearest'}, 'feature_stride_list': [4, 8, 16, 32], 'scale_head_channels': 128,
        'num_fcs': 3, 'coarse_pred_each_layer': True, 'train': {'num_points': 2048, 'oversample_ratio': 3, 'importance_sample_ratio': 0.75},
        'test': {'subdivision_steps': 2, 'subdivision_num_points': 8196, 'scale_factor': 2}, 'dropout': 0,
    },
    'auxiliary': {
        'in_channels': 128, 'dropout': 0, 'num_convs': 0,
    },
    'losses': {
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'mode': 'whole',
        'opts': {}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': False,
        }
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9, 'warmup_cfg': {'type': 'linear', 'ratio': 0.1, 'iters': 200},
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}