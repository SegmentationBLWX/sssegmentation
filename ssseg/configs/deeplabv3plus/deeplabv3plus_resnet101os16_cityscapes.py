'''deeplabv3plus_resnet101os16_cityscapes'''
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
SEGMENTOR_CFG['scheduler']['max_epochs'] = 220
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
    'pretrained': True, 'outstride': 16, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['work_dir'] = 'deeplabv3plus_resnet101os16_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'deeplabv3plus_resnet101os16_cityscapes/deeplabv3plus_resnet101os16_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'deeplabv3plus_resnet101os16_cityscapes/deeplabv3plus_resnet101os16_cityscapes_results.pkl'