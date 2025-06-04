'''SEGMENTOR_CFG for BiSeNetV2'''
from .default_segmentor import SegmentorConfig


'''BISENETV2_SEGMENTOR_CFG'''
BISENETV2_SEGMENTOR_CFG = SegmentorConfig(
    type='FCN',
    num_classes=-1,
    benchmark=True,
    align_corners=False,
    work_dir='ckpts',
    eval_interval_epochs=10,
    save_interval_epochs=1,
    logger_handle_cfg={'type': 'LocalLoggerHandle', 'logfilepath': ''},
    training_logging_manager_cfg={'log_interval_iters': 50},
    norm_cfg={'type': 'SyncBatchNorm'},
    act_cfg={'type': 'ReLU', 'inplace': True},
    backbone={
        'type': 'BiSeNetV2', 'structure_type': 'bisenetv2', 'pretrained': False, 'selected_indices': (0, 1, 2, 3, 4),
    },
    head={
        'in_channels': 128, 'feats_channels': 1024, 'dropout': 0.1, 'num_convs': 1,
    },
    auxiliary=[
        {'in_channels': 16, 'out_channels': 16, 'dropout': 0.1, 'num_convs': 2},
        {'in_channels': 32, 'out_channels': 64, 'dropout': 0.1, 'num_convs': 2},
        {'in_channels': 64, 'out_channels': 256, 'dropout': 0.1, 'num_convs': 2},
        {'in_channels': 128, 'out_channels': 1024, 'dropout': 0.1, 'num_convs': 2},
    ],
    losses={
        'loss_aux1': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_aux2': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_aux3': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_aux4': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
    },
    inference={
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    scheduler={
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9, 'min_lr': 1e-4,
        'warmup_cfg': {'type': 'linear', 'ratio': 0.1, 'iters': 1000},
        'optimizer': {
            'type': 'SGD', 'lr': 0.05, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    dataset=None,
    dataloader=None,
)