'''SEGMENTOR_CFG for MaskFormer'''
SEGMENTOR_CFG = {
    'type': 'maskformer',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'backend': 'nccl',
    'work_dir': 'ckpts',
    'logfilepath': '',
    'log_interval_iterations': 50,
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'resultsavepath': '',
    'norm_cfg': {'type': 'syncbatchnorm'},
    'act_cfg': {'type': 'relu', 'inplace': True},
    'backbone': {
        'type': 'swin_base_patch4_window12_384_22k', 'series': 'swin', 'pretrained': True,
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'layernorm'},
    },
    'head': {
        'in_channels_list': [128, 256, 512, 1024], 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'mask_feats_channels': 256,
        'predictor': {
            'in_channels': None, 'mask_classification': True, 'hidden_dim': 256, 'num_queries': 100, 'nheads': 8, 'num_classes': None,
            'dropout': 0.1, 'dim_feedforward': 2048, 'enc_layers': 0, 'dec_layers': 6, 'pre_norm': False, 'deep_supervision': True,
            'mask_dim': None, 'enforce_input_project': False, 'norm_cfg': {'type': 'layernorm'}, 'act_cfg': {'type': 'relu', 'inplace': True},
        },
        'matcher': {'cost_class': 1.0, 'cost_mask': 20.0, 'cost_dice': 1.0},
    },
    'auxiliary': None,
    'losses': {},
    'inference': {
        'mode': 'whole',
        'opts': {}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': False,
        }
    },
    'scheduler': {
        'type': 'poly', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'adamw', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01,
            'params_rules': {'backbone_net_zerowd': (1.0, 0.0), 'others': (1.0, 1.0)},
        }
    },
    'dataset': None,
    'dataloader': None,
}