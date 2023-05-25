'''SEGMENTOR_CFG for FastSCNN'''
SEGMENTOR_CFG = {
    'type': 'depthwiseseparablefcn',
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
        'type': None, 'series': 'fastscnn', 'pretrained': False, 'selected_indices': (0, 1, 2),
    },
    'head': {
        'in_channels': 128, 'feats_channels': 128, 'dropout': 0.1,
    },
    'auxiliary': [
        {'in_channels': 64, 'out_channels': 32, 'dropout': 0.1,},
        {'in_channels': 128, 'out_channels': 32, 'dropout': 0.1,},
    ],
    'losses': {
        'loss_aux1': {'BinaryCrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_aux2': {'BinaryCrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_cls': {'BinaryCrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    },
    'inference': {
        'mode': 'whole',
        'opts': {}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': False,
        }
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