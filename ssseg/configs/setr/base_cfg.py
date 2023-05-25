'''SEGMENTOR_CFG for SETR'''
SEGMENTOR_CFG = {
    'type': 'setrup',
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
        'type': 'jx_vit_large_p16_384', 'series': 'vit', 'img_size': (512, 512), 'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'layernorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels_list': [1024, 1024, 1024, 1024], 'feats_channels': 256, 'dropout': 0,
        'num_convs': 4, 'scale_factor': 2, 'kernel_size': 3, 'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'auxiliary': [
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 1, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 1, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 1, 'scale_factor': 4, 'kernel_size': 3},
    ],
    'losses': {
        'loss_aux1': {'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_aux2': {'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_aux3': {'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_cls': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    },
    'inference': {
        'mode': 'slide',
        'opts': {'cropsize': (512, 512), 'stride': (341, 341)}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': True,
        }
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0, 'params_rules': {'backbone_net': 0.1, 'others': 1.0},
        }
    },
    'dataset': None,
    'dataloader': None,
}