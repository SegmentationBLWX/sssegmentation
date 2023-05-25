'''ocrnet_hrnetv2w48_ade20k'''
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
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'hrnetv2_w48', 'series': 'hrnet', 'pretrained': True, 'selected_indices': (0, 0),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': sum([48, 96, 192, 384]), 'feats_channels': 512, 'transform_channels': 256, 'scale': 1, 'dropout': 0,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': sum([48, 96, 192, 384]), 'out_channels': 512, 'dropout': 0,
}
SEGMENTOR_CFG['work_dir'] = 'ocrnet_hrnetv2w48_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'ocrnet_hrnetv2w48_ade20k/ocrnet_hrnetv2w48_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'ocrnet_hrnetv2w48_ade20k/ocrnet_hrnetv2w48_ade20k_results.pkl'