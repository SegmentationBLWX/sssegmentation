'''deeplabv3plus_resnet50os8_voc'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_VOCAUG_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_VOCAUG_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 60
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 21
SEGMENTOR_CFG['backbone'] = {
    'type': 'resnet50', 'series': 'resnet', 'pretrained': True,
    'outstride': 8, 'use_stem': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': [256, 2048], 'feats_channels': 512, 'shortcut_channels': 48, 'dilations': [1, 12, 24, 36], 'dropout': 0.1,
}
SEGMENTOR_CFG['work_dir'] = 'deeplabv3plus_resnet50os8_voc'
SEGMENTOR_CFG['logfilepath'] = 'deeplabv3plus_resnet50os8_voc/deeplabv3plus_resnet50os8_voc.log'
SEGMENTOR_CFG['resultsavepath'] = 'deeplabv3plus_resnet50os8_voc/deeplabv3plus_resnet50os8_voc_results.pkl'