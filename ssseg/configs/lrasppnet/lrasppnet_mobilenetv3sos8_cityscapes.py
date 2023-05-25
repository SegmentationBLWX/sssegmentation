'''lrasppnet_mobilenetv3sos8_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 660
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'MobileNetV3', 'structure_type': 'mobilenetv3_small', 'pretrained': True, 'outstride': 8,
    'arch_type': 'small', 'out_indices': (0, 1, 12), 'selected_indices': (0, 1, 2),
}
SEGMENTOR_CFG['head'] = {
    'in_channels_list': [16, 16, 576], 'branch_channels_list': [32, 64], 'feats_channels': 128, 'dropout': 0.1,
}
SEGMENTOR_CFG['work_dir'] = 'lrasppnet_mobilenetv3sos8_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'lrasppnet_mobilenetv3sos8_cityscapes/lrasppnet_mobilenetv3sos8_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'lrasppnet_mobilenetv3sos8_cityscapes/lrasppnet_mobilenetv3sos8_cityscapes_results.pkl'