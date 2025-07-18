'''SEGMENTOR_CFG for IDRNet'''
from .default_segmentor import SegmentorConfig


'''IDRNET_SEGMENTOR_CFG'''
IDRNET_SEGMENTOR_CFG = SegmentorConfig(
    type='IDRNet',
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
        'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
    },
    head={
        'in_channels': 2048, 'feats_channels': 512, 'refine_idcontext_channels': 256, 'refine_coarsecontext_channels': 256,
        'use_sa_on_coarsecontext_before': False, 'use_sa_on_coarsecontext_after': False, 'use_fpn_before': False, 'use_fpn_after': True,
        'force_stage1_use_oripr': False, 'clsrelation_momentum': 0.1, 'dlclsreps_momentum': 0.1, 'ignore_index': -100, 'dropout': 0.1,
    },
    auxiliary={
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    },
    losses={
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls_stage1': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls_stage2': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
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