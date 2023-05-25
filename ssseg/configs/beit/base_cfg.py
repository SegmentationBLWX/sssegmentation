'''SEGMENTOR_CFG for BEiT'''
SEGMENTOR_CFG = {
    'type': 'upernet',
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
        'type': 'beit_base_patch16_224_pt22k_ft22k', 'series': 'beit', 'pretrained': True,
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
    },
    'head': {
        'feature2pyramid': {'embed_dim': 768, 'rescales': [4, 2, 1, 0.5]}, 'in_channels_list': [768, 768, 768, 768], 
        'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 768, 'out_channels': 512, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_cls': {'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    },
    'inference': {
        'mode': 'slide',
        'opts': {'cropsize': (640, 640), 'stride': (426, 426)}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': False,
        }
    },
    'scheduler': {
        'type': 'poly', 'max_epochs': 0, 'power': 1.0, 'min_lr': 0.0, 
        'warmup_cfg': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
        'optimizer': {
            'type': 'adamw', 'lr': 3e-5, 'betas': (0.9, 0.999), 'weight_decay': 0.05, 
            'params_rules': {'type': 'layerdecay', 'num_layers': 12, 'decay_rate': 0.9, 'decay_type': 'layer_wise_vit'},
        }
    },
    'dataset': None,
    'dataloader': None,
}