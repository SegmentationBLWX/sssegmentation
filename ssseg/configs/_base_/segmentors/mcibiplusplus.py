'''SEGMENTOR_CFG for MCIBI++'''
from .default_segmentor import SegmentorConfig


'''MCIBIPLUSPLUS_SEGMENTOR_CFG'''
MCIBIPLUSPLUS_SEGMENTOR_CFG = SegmentorConfig(
    type='MCIBIPlusPlus',
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
        'context_within_image': {
            'is_on': False, 'type': 'aspp',
            'cfg': {'pool_scales': [1, 2, 3, 6], 'dilations': [1, 12, 24, 36]}
        },
        'warmup_epoch': 0, 'use_hard_aggregate': False, 'downsample_before_sa': False,
        'force_use_preds_pr': False, 'fuse_memory_cwi_before_fpn': True, 'in_channels': 2048,
        'feats_channels': 512, 'transform_channels': 256, 'out_channels': 512,
        'update_cfg': {
            'ignore_index': -100,
            'momentum_cfg': {'base_momentum': 0.1, 'base_lr': None, 'adjust_by_learning_rate': False},
        },
        'decoder': {
            'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
            'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
            'cls': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
        },
    },
    auxiliary={'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1},
    losses={
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_pr': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cwi': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
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