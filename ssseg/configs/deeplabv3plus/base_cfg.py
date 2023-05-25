'''SEGMENTOR_CFG for Deeplabv3plus'''
SEGMENTOR_CFG = {
    'benchmark': True,
    'num_classes': -1,
    'align_corners': False,
    'type': 'deeplabv3plus',
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
        'outstride': 8, 'use_stem': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels': [256, 2048], 'feats_channels': 512, 'shortcut_channels': 48, 'dilations': [1, 6, 12, 18], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
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