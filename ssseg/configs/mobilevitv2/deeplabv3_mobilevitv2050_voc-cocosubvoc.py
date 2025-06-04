'''deeplabv3_mobilevitv2050_voc-cocosubvoc'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['MOBILEVITV2_SEGMENTOR_CFG'].copy()
# modify dataset config
DATASET_CFG_VOCAUG_MOBILEVIT_512x512 = REGISTERED_DATASET_CONFIGS['DATASET_CFG_VOCAUG_MOBILEVIT_512x512'].copy()
DATASET_CFG_COCOVOCSUB_MOBILEVIT_512x512 = REGISTERED_DATASET_CONFIGS['DATASET_CFG_VOCAUG_MOBILEVIT_512x512'].copy()
DATASET_CFG_VOCAUG_MOBILEVIT_512x512['seg_target_remapper'] = {i: i for i in range(21)}
DATASET_CFG_COCOVOCSUB_MOBILEVIT_512x512['seg_target_remapper'] = {i: i for i in range(21)}
SEGMENTOR_CFG['dataset'] = {
    'type': 'MultipleDataset',
    'train': {
        'VOCDataset': DATASET_CFG_VOCAUG_MOBILEVIT_512x512.copy(),
        'COCOVOCSUBDataset': DATASET_CFG_COCOVOCSUB_MOBILEVIT_512x512.copy(),
    },
    'test': {
        'VOCDataset': DATASET_CFG_VOCAUG_MOBILEVIT_512x512.copy(),
    }
}
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS64'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 50
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 21
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")