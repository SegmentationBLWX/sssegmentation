'''depthwiseseparablefcn_fastscnn_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS32


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS32.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 1750
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['work_dir'] = 'depthwiseseparablefcn_fastscnn_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'depthwiseseparablefcn_fastscnn_cityscapes/depthwiseseparablefcn_fastscnn_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'depthwiseseparablefcn_fastscnn_cityscapes/depthwiseseparablefcn_fastscnn_cityscapes_results.pkl'