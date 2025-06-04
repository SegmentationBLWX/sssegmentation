'''SEGMENTOR_CFG for OCRNet'''
from .default_segmentor import SegmentorConfig


'''OCRNET_SEGMENTOR_CFG'''
OCRNET_SEGMENTOR_CFG = SegmentorConfig(
    type='OCRNet',
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
        'type': 'HRNet', 'structure_type': 'hrnetv2_w18', 'arch': 'hrnetv2_w18', 'pretrained': True, 'selected_indices': (0, 0),
    },
    head={
        'in_channels': sum([18, 36, 72, 144]), 'feats_channels': 512, 'transform_channels': 256, 'scale': 1, 'dropout': 0,
    },
    auxiliary={
        'in_channels': sum([18, 36, 72, 144]), 'out_channels': 512, 'dropout': 0,
    },
    losses={
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
    },
    inference={
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    scheduler={
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    dataset=None,
    dataloader=None,
)