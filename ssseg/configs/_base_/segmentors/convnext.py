'''SEGMENTOR_CFG for ConvNeXt'''
from .default_segmentor import SegmentorConfig


'''CONVNEXT_SEGMENTOR_CFG'''
CONVNEXT_SEGMENTOR_CFG = SegmentorConfig(
    type='UPerNet',
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
        'type': 'ConvNeXt', 'structure_type': 'convnext_base', 'arch': 'base', 'pretrained': True, 'drop_path_rate': 0.4,
        'layer_scale_init_value': 1.0, 'gap_before_final_norm': False, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm2d', 'eps': 1e-6},
    },
    head={
        'in_channels_list': [128, 256, 512, 1024], 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    auxiliary={
        'in_channels': 512, 'out_channels': 512, 'dropout': 0.1,
    },
    losses={
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
    },
    inference={
        'forward': {'mode': 'slide', 'cropsize': (512, 512), 'stride': (341, 341)},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    scheduler={
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'AdamW', 'lr': 0.0001, 'betas': (0.9, 0.999), 'weight_decay': 0.05,
            'params_rules': {'type': 'LearningRateDecayParamsConstructor', 'decay_rate': 0.9, 'decay_type': 'stage_wise', 'num_layers': 12},
        }
    },
    dataset=None,
    dataloader=None,
)