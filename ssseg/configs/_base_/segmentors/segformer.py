'''SEGMENTOR_CFG for Segformer'''
from .default_segmentor import SegmentorConfig


'''SEGFORMER_SEGMENTOR_CFG'''
SEGFORMER_SEGMENTOR_CFG = SegmentorConfig(
    type='Segformer',
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
        'type': 'MixVisionTransformer', 'structure_type': 'mit-b0', 'pretrained': True, 'pretrained_model_path': 'mit_b0.pth',
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
        'embed_dims': 32, 'num_stages': 4, 'num_layers': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8], 'patch_sizes': [7, 3, 3, 3],
        'sr_ratios': [8, 4, 2, 1], 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.1,
    },
    head={
        'in_channels_list': [32, 64, 160, 256], 'feats_channels': 256, 'dropout': 0.1,
    },
    losses={
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    inference={
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': True},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    scheduler={
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 1.0, 'min_lr': 0.0, 'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
        'optimizer': {
            'type': 'AdamW', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
            'params_rules': {
                'norm': dict(wd_multiplier=0.),
                'convs': dict(lr_multiplier=10.0, wd_multiplier=1.0),
                'decoder': dict(lr_multiplier=10.0, wd_multiplier=1.0),
            },
        }
    },
    dataset=None,
    dataloader=None,
)