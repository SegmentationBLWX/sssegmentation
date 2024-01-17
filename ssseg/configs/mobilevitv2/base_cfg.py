'''SEGMENTOR_CFG for MobileViTV2'''
SEGMENTOR_CFG = {
    'benchmark': True,
    'num_classes': -1,
    'align_corners': False,
    'type': 'Deeplabv3',
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
        'type': 'MobileViTV2', 'structure_type': 'mobilevitv2_050', 'pretrained': True, 'selected_indices': (3, 4),
    },
    'head': {
        'in_channels': 256, 'feats_channels': 512, 'dilations': [1, 6, 12, 18], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 192, 'out_channels': 512, 'dropout': 0.1,
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
        'type': 'CosineScheduler', 'max_epochs': 0, 'by_epoch': True, 'min_lr': 1.e-6, 'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 500},
        'optimizer': {
            'type': 'AdamW', 'lr': 0.0005, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
            'params_rules': {
                'norm': dict(wd_multiplier=0.),
            },
        }
    },
    'dataset': None,
    'dataloader': None,
}