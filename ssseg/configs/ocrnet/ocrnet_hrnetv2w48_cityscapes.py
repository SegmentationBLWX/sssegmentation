'''ocrnet_hrnetv2w48_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS8


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS8.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 440
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'hrnetv2_w48', 'series': 'hrnet', 'pretrained': True, 'selected_indices': (0, 0),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': sum([48, 96, 192, 384]), 'feats_channels': 512, 'transform_channels': 256, 'scale': 1, 'dropout': 0,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': sum([48, 96, 192, 384]), 'out_channels': 512, 'dropout': 0,
}
SEGMENTOR_CFG['work_dir'] = 'ocrnet_hrnetv2w48_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'ocrnet_hrnetv2w48_cityscapes/ocrnet_hrnetv2w48_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'ocrnet_hrnetv2w48_cityscapes/ocrnet_hrnetv2w48_cityscapes_results.pkl'