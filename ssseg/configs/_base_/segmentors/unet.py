'''SEGMENTOR_CFG for UNet'''
from .default_segmentor import SegmentorConfig


'''UNET_SEGMENTOR_CFG'''
UNET_SEGMENTOR_CFG = {
    'type': 'FCN',
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
        'type': 'UNet', 'structure_type': 'unets5os16', 'pretrained': False, 'selected_indices': (3, 4),
    },
    'head': {
        'in_channels': 64, 'feats_channels': 64, 'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 128, 'out_channels': 64, 'dropout': 0.1,
    },
    'losses': {
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    },
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['dice', 'mdice']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    'dataset': None,
    'dataloader': None,
}