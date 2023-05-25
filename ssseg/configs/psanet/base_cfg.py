'''SEGMENTOR_CFG for PSANet'''
SEGMENTOR_CFG = {
    'type': 'psanet',
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
    'norm_cfg': {'type': 'syncbatchnorm'},
    'act_cfg': {'type': 'relu', 'inplace': True},
    'backbone': {
        'type': 'resnet101', 'series': 'resnet', 'pretrained': True,
        'outstride': 8, 'use_stem': True, 'selected_indices': (2, 3),
    },
    'head': {
        'in_channels': 2048, 'feats_channels': 512, 'type': 'bi-direction',
        'mask_size': (97, 97), 'compact': False, 'shrink_factor': 2,
        'normalization_factor': 1.0, 'psa_softmax': True, 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_cls': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    },
    'inference': {
        'mode': 'whole',
        'opts': {}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': False,
        }
    },
    'scheduler': {
        'type': 'poly', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}