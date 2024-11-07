'''SEGMENTOR_CFG for SemanticFPN'''
SEGMENTOR_CFG = {
    'type': 'SemanticFPN',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'init_process_group_cfg': {'backend': 'nccl', 'timeout': 7200},
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 50},
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
        'pretrained': True, 'outstride': 32, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels_list': [256, 512, 1024, 2048], 'feats_channels': 256, 'upsample_cfg': {'mode': 'nearest'},
        'feature_stride_list': [4, 8, 16, 32], 'scale_head_channels': 128, 'dropout': 0.1,
    },
    'losses': {
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}