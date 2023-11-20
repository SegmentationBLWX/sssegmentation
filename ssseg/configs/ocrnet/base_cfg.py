'''SEGMENTOR_CFG for OCRNet'''
SEGMENTOR_CFG = {
    'type': 'OCRNet',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'backend': 'nccl',
    'work_dir': 'ckpts',
    'logfilepath': '',
    'log_interval_iterations': 50,
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'resultsavepath': '',
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'HRNet', 'structure_type': 'hrnetv2_w18', 'arch': 'hrnetv2_w18', 'pretrained': True, 'selected_indices': (0, 0),
    },
    'head': {
        'in_channels': sum([18, 36, 72, 144]), 'feats_channels': 512, 'transform_channels': 256, 'scale': 1, 'dropout': 0,
    },
    'auxiliary': {
        'in_channels': sum([18, 36, 72, 144]), 'out_channels': 512, 'dropout': 0,
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
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}