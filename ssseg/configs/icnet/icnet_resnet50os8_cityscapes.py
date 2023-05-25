'''icnet_resnet50os8_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_832x832, DATALOADER_CFG_BS8


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_832x832.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS8.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 440
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2),
}
SEGMENTOR_CFG['work_dir'] = 'icnet_resnet50os8_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'icnet_resnet50os8_cityscapes/icnet_resnet50os8_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'icnet_resnet50os8_cityscapes/icnet_resnet50os8_cityscapes_results.pkl'