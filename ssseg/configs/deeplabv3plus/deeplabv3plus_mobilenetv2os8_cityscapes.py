'''deeplabv3plus_mobilenetv2os8_cityscapes'''
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
    'type': 'MobileNetV2', 'structure_type': 'mobilenetv2', 'pretrained': True, 'outstride': 8, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': [24, 320], 'feats_channels': 512, 'shortcut_channels': 48, 'dilations': [1, 12, 24, 36], 'dropout': 0.1,
}
SEGMENTOR_CFG['auxiliary'] = {
    'in_channels': 96, 'out_channels': 512, 'dropout': 0.1,
}
SEGMENTOR_CFG['work_dir'] = 'deeplabv3plus_mobilenetv2os8_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'deeplabv3plus_mobilenetv2os8_cityscapes/deeplabv3plus_mobilenetv2os8_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'deeplabv3plus_mobilenetv2os8_cityscapes/deeplabv3plus_mobilenetv2os8_cityscapes_results.pkl'