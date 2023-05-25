'''segformer_mitb5_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'mit-b5', 'series': 'mit', 'pretrained': True, 'pretrained_model_path': 'mit_b5.pth', 
    'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'layernorm', 'eps': 1e-6},
}
SEGMENTOR_CFG['head'] = {
    'in_channels_list': [64, 128, 320, 512], 'feats_channels': 256, 'dropout': 0.1,
}
SEGMENTOR_CFG['work_dir'] = 'segformer_mitb5_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'segformer_mitb5_ade20k/segformer_mitb5_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'segformer_mitb5_ade20k/segformer_mitb5_ade20k_results.pkl'