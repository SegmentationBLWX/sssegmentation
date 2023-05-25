'''lrasppnet_mobilenetv3los8_cityscapes'''
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
SEGMENTOR_CFG['work_dir'] = 'lrasppnet_mobilenetv3los8_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'lrasppnet_mobilenetv3los8_cityscapes/lrasppnet_mobilenetv3los8_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'lrasppnet_mobilenetv3los8_cityscapes/lrasppnet_mobilenetv3los8_cityscapes_results.pkl'