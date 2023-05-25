'''isnet_resnet101os8_lip'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_LIP_473x473, DATALOADER_CFG_BS32


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_LIP_473x473.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS32.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 150
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 20
SEGMENTOR_CFG['act_cfg'] = {'type': 'leakyrelu', 'negative_slope': 0.01, 'inplace': True}
SEGMENTOR_CFG['head']['shortcut']['is_on'] = True
SEGMENTOR_CFG['work_dir'] = 'isnet_resnet101os8_lip'
SEGMENTOR_CFG['logfilepath'] = 'isnet_resnet101os8_lip/isnet_resnet101os8_lip.log'
SEGMENTOR_CFG['resultsavepath'] = 'isnet_resnet101os8_lip/isnet_resnet101os8_lip_results.pkl'