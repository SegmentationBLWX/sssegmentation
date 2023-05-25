'''ocrnet_hrnetv2w18s_cityscapes'''
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
    'type': 'HRNet', 'structure_type': 'hrnetv2_w18_small', 'arch': 'hrnetv2_w18_small', 'pretrained': True, 'selected_indices': (0, 0),
}
SEGMENTOR_CFG['work_dir'] = 'ocrnet_hrnetv2w18s_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'ocrnet_hrnetv2w18s_cityscapes/ocrnet_hrnetv2w18s_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'ocrnet_hrnetv2w18s_cityscapes/ocrnet_hrnetv2w18s_cityscapes_results.pkl'