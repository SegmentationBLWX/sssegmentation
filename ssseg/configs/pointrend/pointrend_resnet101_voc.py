'''pointrend_resnet101_voc'''
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
SEGMENTOR_CFG['work_dir'] = 'pointrend_resnet101_voc'
SEGMENTOR_CFG['logfilepath'] = 'pointrend_resnet101_voc/pointrend_resnet101_voc.log'
SEGMENTOR_CFG['resultsavepath'] = 'pointrend_resnet101_voc/pointrend_resnet101_voc_results.pkl'