'''SEGMENTOR_CFG for LRASPPNet'''
SEGMENTOR_CFG = {
    'type': 'LRASPPNet',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 50},
    'norm_cfg': {'type': 'SyncBatchNorm', 'eps': 0.001},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'MobileNetV3', 'structure_type': 'mobilenetv3_large', 'pretrained': True,
        'outstride': 8, 'arch_type': 'large', 'selected_indices': (0, 1, 2),
    },
    'head': {
        'in_channels_list': [16, 24, 960], 'branch_channels_list': [32, 64], 'feats_channels': 128, 'dropout': 0.1,
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