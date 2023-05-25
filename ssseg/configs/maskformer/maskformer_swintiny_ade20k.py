'''maskformer_swintiny_ade20k'''
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
SEGMENTOR_CFG['scheduler']['min_lr'] = 0.0
SEGMENTOR_CFG['scheduler']['power'] = 1.0
SEGMENTOR_CFG['scheduler']['warmup_cfg'] = {'type': 'linear', 'ratio': 1e-6, 'iters': 1500}
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'swin_tiny_patch4_window7_224', 'series': 'swin', 'pretrained': True, 'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'layernorm'},
}
SEGMENTOR_CFG['head']['in_channels_list'] = [96, 192, 384, 768]
SEGMENTOR_CFG['work_dir'] = 'maskformer_swintiny_ade20k'
SEGMENTOR_CFG['logfilepath'] = 'maskformer_swintiny_ade20k/maskformer_swintiny_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'maskformer_swintiny_ade20k/maskformer_swintiny_ade20k_results.pkl'