'''SEGMENTOR_CFG for CGNet'''
SEGMENTOR_CFG = {
    'type': 'FCN',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 50},
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'CGNet', 'structure_type': 'cgnetm3n21', 'pretrained': False, 'selected_indices': (1, 2),
    },
    'head': {
        'in_channels': 256, 'feats_channels': 256, 'dropout': 0, 'num_convs': 0,
    },
    'auxiliary': None,
    'losses': {
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9, 'min_lr': 1e-4,
        'optimizer': {
            'type': 'Adam', 'lr': 0.001, 'eps': 1e-08, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}