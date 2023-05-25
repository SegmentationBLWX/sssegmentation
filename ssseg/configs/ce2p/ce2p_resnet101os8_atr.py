'''ce2p_resnet101os8_atr'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ATR_473x473, DATALOADER_CFG_BS32


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ATR_473x473.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS32.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 150
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 18
SEGMENTOR_CFG['work_dir'] = 'ce2p_resnet101os8_atr'
SEGMENTOR_CFG['logfilepath'] = 'ce2p_resnet101os8_atr/ce2p_resnet101os8_atr.log'
SEGMENTOR_CFG['resultsavepath'] = 'ce2p_resnet101os8_atr/ce2p_resnet101os8_atr_results.pkl'