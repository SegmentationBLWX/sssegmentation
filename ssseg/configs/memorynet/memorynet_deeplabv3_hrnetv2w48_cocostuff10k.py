'''memorynet_deeplabv3_hrnetv2w48_cocostuff10k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_COCOStuff10k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_COCOStuff10k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 110
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 182
SEGMENTOR_CFG['backbone'] = {
    'type': 'hrnetv2_w48', 'series': 'hrnet', 'pretrained': True, 'selected_indices': (0, 0, 0, 0),
}
SEGMENTOR_CFG['head']['use_loss'] = False
SEGMENTOR_CFG['head']['in_channels'] = sum([48, 96, 192, 384])
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.001 * 0.9
SEGMENTOR_CFG['auxiliary'] = None
SEGMENTOR_CFG['losses'].pop('loss_aux')
SEGMENTOR_CFG['work_dir'] = 'memorynet_deeplabv3_hrnetv2w48_cocostuff10k'
SEGMENTOR_CFG['logfilepath'] = 'memorynet_deeplabv3_hrnetv2w48_cocostuff10k/memorynet_deeplabv3_hrnetv2w48_cocostuff10k.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynet_deeplabv3_hrnetv2w48_cocostuff10k/memorynet_deeplabv3_hrnetv2w48_cocostuff10k_results.pkl'