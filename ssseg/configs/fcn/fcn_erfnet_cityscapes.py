'''fcn_erfnet_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_1024x1024, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_1024x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 860
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': None, 'series': 'erfnet', 'pretrained': False, 'selected_indices': (0,),
}
SEGMENTOR_CFG['head'] = {
    'in_channels': 16, 'feats_channels': 128, 'dropout': 0.1, 'num_convs': 1,
}
SEGMENTOR_CFG['auxiliary'] = None
SEGMENTOR_CFG['backbone']['losses'].pop('loss_aux')
SEGMENTOR_CFG['work_dir'] = 'fcn_erfnet_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'fcn_erfnet_cityscapes/fcn_erfnet_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'fcn_erfnet_cityscapes/fcn_erfnet_cityscapes_results.pkl'