'''mcibi_deeplabv3_resnet50os8_lip'''
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
SEGMENTOR_CFG['act_cfg'] = {'type': 'LeakyReLU', 'negative_slope': 0.01, 'inplace': True}
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
}
SEGMENTOR_CFG['head']['use_loss'] = False
SEGMENTOR_CFG['work_dir'] = 'mcibi_deeplabv3_resnet50os8_lip'
SEGMENTOR_CFG['logfilepath'] = 'mcibi_deeplabv3_resnet50os8_lip/mcibi_deeplabv3_resnet50os8_lip.log'
SEGMENTOR_CFG['resultsavepath'] = 'mcibi_deeplabv3_resnet50os8_lip/mcibi_deeplabv3_resnet50os8_lip_results.pkl'