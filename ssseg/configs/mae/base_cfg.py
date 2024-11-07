'''SEGMENTOR_CFG for MAE'''
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
        'type': 'MAE', 'structure_type': 'mae_pretrain_vit_base', 'pretrained': True, 
        'img_size': (512, 512), 'patch_size': 16, 'embed_dims': 768, 'num_layers': 12,
        'num_heads': 12, 'mlp_ratio': 4, 'init_values': 1.0, 'drop_path_rate': 0.1,
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
    },
    'head': {
        'feature2pyramid': {'embed_dim': 768, 'rescales': [4, 2, 1, 0.5]}, 'in_channels_list': [768, 768, 768, 768], 
        'feats_channels': 768, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 768, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'slide', 'cropsize': (512, 512), 'stride': (341, 341)},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 1.0, 'min_lr': 0.0, 
        'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
        'optimizer': {
            'type': 'AdamW', 'lr': 1e-4, 'betas': (0.9, 0.999), 'weight_decay': 0.05, 
            'params_rules': {'type': 'LearningRateDecayParamsConstructor', 'num_layers': 12, 'decay_rate': 0.65, 'decay_type': 'layer_wise_vit'},
        }
    },
    'dataset': None,
    'dataloader': None,
}