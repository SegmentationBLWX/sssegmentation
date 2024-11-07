'''SEGMENTOR_CFG for SwinTransformer'''
SEGMENTOR_CFG = {
    'type': 'UPerNet',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 50},
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'SwinTransformer', 'structure_type': 'swin_base_patch4_window12_384_22k', 'pretrained': True,
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
        'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 128, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
        'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
        'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,
    },
    'head': {
        'in_channels_list': [128, 256, 512, 1024], 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 512, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 1.0, 'min_lr': 0.0,
        'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
        'optimizer': {
            'type': 'AdamW', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
            'params_rules': {
                'norm': dict(wd_multiplier=0.),
                'position_encodings': dict(wd_multiplier=0.),
            },
        }
    },
    'dataset': None,
    'dataloader': None,
}