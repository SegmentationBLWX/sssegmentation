'''SEGMENTOR_CFG for SegFormer'''
SEGMENTOR_CFG = {
    'type': 'segformer',
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
        'type': 'mit-b0', 'series': 'mit', 'pretrained': True, 'pretrained_model_path': 'mit_b0.pth',
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'head': {
        'in_channels_list': [32, 64, 160, 256], 'feats_channels': 256, 'dropout': 0.1,
    },
    'losses': {
        'loss_cls': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    },
    'inference': {
        'mode': 'whole',
        'opts': {}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': True,
        }
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 1.0, 'min_lr': 0.0, 'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
        'optimizer': {
            'type': 'adamw', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
            'params_rules': {'backbone_net_zerowd': (1.0, 0.0), 'backbone_net_nonzerowd': (1.0, 1.0), 'others': (10.0, 1.0)},
        }
    },
    'dataset': None,
    'dataloader': None,
}