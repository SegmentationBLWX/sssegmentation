'''SEGMENTOR_CFG for PointRend'''
SEGMENTOR_CFG = {
    'type': 'pointrend',
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
        'type': 'resnet101', 'series': 'resnet', 'pretrained': True, 'outstride': 32,
        'use_stem': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'fpn_in_channels_list': [256, 512, 1024, 2048], 'pointrend_in_channels_list': [256], 'feats_channels': 256,
        'upsample_cfg': {'mode': 'nearest'}, 'feature_stride_list': [4, 8, 16, 32], 'scale_head_channels': 128,
        'num_fcs': 3, 'coarse_pred_each_layer': True, 'train': {'num_points': 2048, 'oversample_ratio': 3, 'importance_sample_ratio': 0.75},
        'test': {'subdivision_steps': 2, 'subdivision_num_points': 8196, 'scale_factor': 2}, 'dropout': 0,
    },
    'auxiliary': {
        'in_channels': 128, 'dropout': 0, 'num_convs': 0,
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
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9, 'warmup_cfg': {'type': 'linear', 'ratio': 0.1, 'iters': 200},
        'optimizer': {
            'type': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}