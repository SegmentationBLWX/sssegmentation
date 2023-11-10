'''mcibi_deeplabv3_hrnetv2w48_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 180
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.004, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'HRNet', 'structure_type': 'hrnetv2_w48', 'arch': 'hrnetv2_w48', 'pretrained': True, 'selected_indices': (0, 0, 0, 0),
}
SEGMENTOR_CFG['head']['in_channels'] = sum([48, 96, 192, 384])
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.004
SEGMENTOR_CFG['auxiliary'] = None
SEGMENTOR_CFG['losses'].pop('loss_aux')
SEGMENTOR_CFG['work_dir'] = 'mcibi_deeplabv3_hrnetv2w48_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'mcibi_deeplabv3_hrnetv2w48_ade20k/mcibi_deeplabv3_hrnetv2w48_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibi_deeplabv3_hrnetv2w48_ade20k/mcibi_deeplabv3_hrnetv2w48_ade20k_results.pkl'