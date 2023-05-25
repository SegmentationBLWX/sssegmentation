'''ocrnet_resnet101os8_cityscapes'''
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
    'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 2048, 'feats_channels': 512, 'transform_channels': 256, 'scale': 1, 'dropout': 0,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 1024, 'out_channels': 512, 'dropout': 0,
}
SEGMENTOR_CFG['work_dir'] = 'ocrnet_resnet101os8_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'ocrnet_resnet101os8_cityscapes/ocrnet_resnet101os8_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'ocrnet_resnet101os8_cityscapes/ocrnet_resnet101os8_cityscapes_results.pkl'