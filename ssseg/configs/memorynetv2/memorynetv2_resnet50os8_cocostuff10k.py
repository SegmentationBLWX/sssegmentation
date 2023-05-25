'''memorynetv2_resnet50os8_cocostuff10k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_COCOStuff10k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_COCOStuff10k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 110
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4, 'params_rules': {},
}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 182
SEGMENTOR_CFG['backbone'] = {
    'type': 'resnet50', 'series': 'resnet', 'pretrained': True,
    'outstride': 8, 'use_stem': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['losses'].pop('loss_cwi')
SEGMENTOR_CFG['work_dir'] = 'memorynetv2_resnet50os8_cocostuff10k'
SEGMENTOR_CFG['logfilepath'] = 'memorynetv2_resnet50os8_cocostuff10k/memorynetv2_resnet50os8_cocostuff10k.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynetv2_resnet50os8_cocostuff10k/memorynetv2_resnet50os8_cocostuff10k_results.pkl'