'''SEGMENTOR_CFG for SemanticFPN'''
SEGMENTOR_CFG = {
    'type': 'semanticfpn',
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
        'type': 'resnet101', 'series': 'resnet', 'pretrained': True,
        'outstride': 32, 'use_stem': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels_list': [256, 512, 1024, 2048], 'feats_channels': 256, 'upsample_cfg': {'mode': 'nearest'},
        'feature_stride_list': [4, 8, 16, 32], 'scale_head_channels': 128, 'dropout': 0.1,
    },
    'losses': {
        'loss_cls': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
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