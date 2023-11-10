'''mcibiplusplus_aspp_hrnetv2w48_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
SEGMENTOR_CFG['dataset']['train']['set'] = 'trainval'
SEGMENTOR_CFG['dataset']['test']['set'] = 'test'
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 440
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'HRNet', 'structure_type': 'hrnetv2_w48', 'arch': 'hrnetv2_w48', 'pretrained': True, 'selected_indices': (0, 0, 0, 0),
}
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
    'cls': {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
}
SEGMENTOR_CFG['head']['downsample_before_sa'] = True
SEGMENTOR_CFG['head']['in_channels'] = sum([48, 96, 192, 384])
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['head']['context_within_image']['use_self_attention'] = False
SEGMENTOR_CFG['auxiliary'] = None
SEGMENTOR_CFG['losses'].pop('loss_aux')
SEGMENTOR_CFG['inference'] = {
    'mode': 'slide',
    'opts': {'cropsize': (1024, 512), 'stride': (341, 341)}, 
    'tricks': {
        'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        'flip': True,
        'use_probs_before_resize': True,
    }
}
SEGMENTOR_CFG['work_dir'] = 'mcibiplusplus_aspp_hrnetv2w48_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'mcibiplusplus_aspp_hrnetv2w48_cityscapes/mcibiplusplus_aspp_hrnetv2w48_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibiplusplus_aspp_hrnetv2w48_cityscapes/mcibiplusplus_aspp_hrnetv2w48_cityscapes_results.pkl'