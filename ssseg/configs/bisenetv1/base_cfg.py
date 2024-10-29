'''SEGMENTOR_CFG for BiSeNetV1'''
SEGMENTOR_CFG = {
    'type': 'FCN',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
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
        'type': 'BiSeNetV1', 'structure_type': 'bisenetv1', 'pretrained': False, 'selected_indices': (0, 1, 2),
        'spatial_channels_list': (256, 256, 256, 512), 'context_channels_list': (512, 1024, 2048), 'out_channels': 1024,
        'backbone_cfg': {'type': 'ResNet', 'structure_type': 'resnet50conv3x3stem', 'depth': 50, 'pretrained': True, 'outstride': 32, 'use_conv3x3_stem': True},
    },
    'head': {
        'in_channels': 1024, 'feats_channels': 1024, 'dropout': 0.1, 'num_convs': 1,
    },
    'auxiliary': [
        {'in_channels': 512, 'out_channels': 256, 'dropout': 0.1, 'num_convs': 1},
        {'in_channels': 512, 'out_channels': 256, 'dropout': 0.1, 'num_convs': 1},
    ],
    'losses': {
        'loss_aux1': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_aux2': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9, 'min_lr': 1e-4,
        'warmup_cfg': {'type': 'linear', 'ratio': 0.1, 'iters': 1000},
        'optimizer': {
            'type': 'SGD', 'lr': 0.05, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}