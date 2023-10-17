'''idrnet_aspp_resnet50os8_lip'''
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
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
SEGMENTOR_CFG['head']['use_fpn_before'] = False
SEGMENTOR_CFG['head']['use_fpn_after'] = True
SEGMENTOR_CFG['head']['use_sa_on_coarsecontext_before'] = False
SEGMENTOR_CFG['head']['use_sa_on_coarsecontext_after'] = False
SEGMENTOR_CFG['head']['coarse_context'] = {'type': 'aspp', 'dilations': (1, 12, 24, 36)}
SEGMENTOR_CFG['work_dir'] = 'idrnet_aspp_resnet50os8_lip'
SEGMENTOR_CFG['logfilepath'] = 'idrnet_aspp_resnet50os8_lip/idrnet_aspp_resnet50os8_lip.log'
SEGMENTOR_CFG['resultsavepath'] = 'idrnet_aspp_resnet50os8_lip/idrnet_aspp_resnet50os8_lip_results.pkl'