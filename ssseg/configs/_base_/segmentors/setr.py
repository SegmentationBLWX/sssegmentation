'''SEGMENTOR_CFG for SETR'''
from .default_segmentor import SegmentorConfig


'''SETR_SEGMENTOR_CFG'''
SETR_SEGMENTOR_CFG = SegmentorConfig(
    type='SETRUP',
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
        'type': 'VisionTransformer', 'structure_type': 'jx_vit_large_p16_384', 'img_size': (512, 512), 'out_indices': (9, 14, 19, 23),
        'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
        'patch_size': 16, 'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
        'qkv_bias': True, 'drop_rate': 0.1, 'attn_drop_rate': 0., 'drop_path_rate': 0., 'with_cls_token': True,
        'output_cls_token': False, 'patch_norm': False, 'final_norm': False, 'num_fcs': 2,
    },
    head={
        'in_channels_list': [1024, 1024, 1024, 1024], 'feats_channels': 256, 'dropout': 0,
        'num_convs': 4, 'scale_factor': 2, 'kernel_size': 3, 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
    },
    auxiliary=[
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 1, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 1, 'scale_factor': 4, 'kernel_size': 3},
        {'in_channels': 1024, 'out_channels': 256, 'dropout': 0, 'num_convs': 1, 'scale_factor': 4, 'kernel_size': 3},
    ],
    losses={
        'loss_aux1': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_aux2': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_aux3': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
    },
    inference={
        'forward': {'mode': 'slide', 'cropsize': (512, 512), 'stride': (341, 341)},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': True},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    scheduler={
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0,
            'params_rules': {
                'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0),
            },
        }
    },
    dataset=None,
    dataloader=None,
)