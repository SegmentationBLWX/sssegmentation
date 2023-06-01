'''upernet_beitbase_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_640x640, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_640x640.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['inference']['tricks']['use_probs_before_resize'] = True
SEGMENTOR_CFG['work_dir'] = 'upernet_beitbase_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'upernet_beitbase_ade20k/upernet_beitbase_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'upernet_beitbase_ade20k/upernet_beitbase_ade20k_results.pkl'