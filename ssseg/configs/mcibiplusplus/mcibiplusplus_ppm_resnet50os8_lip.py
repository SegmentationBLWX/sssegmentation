'''mcibiplusplus_ppm_resnet50os8_lip'''
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
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head']['decoder'] = {
    'pr': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cwi': {'in_channels': 512, 'out_channels': 512, 'dropout': 0.1},
    'cls': {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1},
}
SEGMENTOR_CFG['head']['context_within_image']['type'] = 'ppm'
SEGMENTOR_CFG['head']['context_within_image']['is_on'] = True
SEGMENTOR_CFG['work_dir'] = 'mcibiplusplus_ppm_resnet50os8_lip'
SEGMENTOR_CFG['logfilepath'] = 'mcibiplusplus_ppm_resnet50os8_lip/mcibiplusplus_ppm_resnet50os8_lip.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibiplusplus_ppm_resnet50os8_lip/mcibiplusplus_ppm_resnet50os8_lip_results.pkl'