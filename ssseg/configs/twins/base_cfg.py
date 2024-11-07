'''SEGMENTOR_CFG for Twins'''
SEGMENTOR_CFG = {
    'type': 'UPerNet',
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
        'type': 'PCPVT', 'structure_type': 'pcpvt_base', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 
        'norm_cfg': {'type': 'LayerNorm'}, 'depths': [3, 4, 18, 3], 'drop_path_rate': 0.3,
    },
    'head': {
        'in_channels_list': [64, 128, 320, 512], 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 320, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 1.0, 'min_lr': 0.0,
        'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
        'optimizer': {
            'type': 'AdamW', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
            'params_rules': {
                'norm': dict(wd_multiplier=0.),
                'position_encodings': dict(wd_multiplier=0.),
            },
        }
    },
    'dataset': None,
    'dataloader': None,
}