'''SEGMENTOR_CFG for FastSCNN'''
SEGMENTOR_CFG = {
    'type': 'DepthwiseSeparableFCN',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'backend': 'nccl',
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 50},
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'FastSCNN', 'structure_type': 'fastscnn', 'pretrained': False, 'selected_indices': (0, 1, 2),
    },
    'head': {
        'in_channels': 128, 'feats_channels': 128, 'dropout': 0.1,
    },
    'auxiliary': [
        {'in_channels': 64, 'out_channels': 32, 'dropout': 0.1,},
        {'in_channels': 128, 'out_channels': 32, 'dropout': 0.1,},
    ],
    'losses': {
        'loss_aux1': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean', 'use_sigmoid': True},
        'loss_aux2': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean', 'use_sigmoid': True},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean', 'use_sigmoid': True},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9, 'min_lr': 1e-4,
        'optimizer': {
            'type': 'SGD', 'lr': 0.12, 'momentum': 0.9, 'weight_decay': 4e-5, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}