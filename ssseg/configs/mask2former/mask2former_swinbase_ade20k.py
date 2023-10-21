'''mask2former_swinbase_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_640x640, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_640x640.copy()
SEGMENTOR_CFG['dataset']['train']['data_pipelines'][0] = (
    'RandomChoiceResize', {
        'scales': [int(512 * x * 0.1) for x in range(5, 21)], 
        'resize_type': 'ResizeShortestEdge', 
        'max_size': 2048,
    }
)
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
SEGMENTOR_CFG['scheduler']['min_lr'] = 0.0
SEGMENTOR_CFG['scheduler']['clipgrad_cfg'] = {'max_norm': 0.01, 'norm_type': 2}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['work_dir'] = 'mask2former_swinbase_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'mask2former_swinbase_ade20k/mask2former_swinbase_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'mask2former_swinbase_ade20k/mask2former_swinbase_ade20k_results.pkl'