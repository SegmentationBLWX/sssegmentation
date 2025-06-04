'''mask2former_swinbase_ade20k'''
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['MASK2FORMER_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_ADE20k_640x640'].copy()
SEGMENTOR_CFG['dataset']['train']['data_pipelines'][0] = (
    'RandomChoiceResize', {
        'scales': [int(512 * x * 0.1) for x in range(5, 21)], 
        'resize_type': 'ResizeShortestEdge', 
        'max_size': 2048,
    }
)
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
SEGMENTOR_CFG['scheduler']['min_lr'] = 0.0
SEGMENTOR_CFG['scheduler']['clipgrad_cfg'] = {'max_norm': 0.01, 'norm_type': 2}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")