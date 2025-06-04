'''SEGMENTOR_CFG for MCIBI'''
from .default_segmentor import SegmentorConfig


'''MCIBI_SEGMENTOR_CFG'''
MCIBI_SEGMENTOR_CFG = SegmentorConfig(
    type='MCIBI',
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
        'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
        'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
    },
    head={
        'downsample_backbone': {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
        'context_within_image': {'is_on': True, 'type': 'aspp', 'cfg': {'dilations': [1, 12, 24, 36]}},
        'use_hard_aggregate': False, 'in_channels': 2048, 'feats_channels': 512, 'transform_channels': 256,
        'out_channels': 512, 'num_feats_per_cls': 1, 'use_loss': True, 'loss_cfg': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'reduction': 'mean'},
        'update_cfg': {
            'strategy': 'cosine_similarity', 'ignore_index': 255,
            'momentum_cfg': {'base_momentum': 0.9, 'base_lr': 0.01, 'adjust_by_learning_rate': True}
        },
        'dropout': 0.1,
    },
    auxiliary={
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    },
    losses={
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls_stage1': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls_stage2': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
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