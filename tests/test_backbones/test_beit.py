'''
Function:
    Implementation of Testing BEiT
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.beit import DEFAULT_MODEL_URLS


'''BEiTs'''
cfgs = [
    {'type': 'BEiT', 'structure_type': 'beit_base_patch16_224_pt22k_ft22k', 'pretrained': True, 
     'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},},
    {'type': 'BEiT', 'structure_type': 'beit_large_patch16_224_pt22k_ft22k', 'pretrained': True, 
     'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
     'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
     'qv_bias': True, 'init_values': 1e-6, 'drop_path_rate': 0.2, 'out_indices': [7, 11, 15, 23]},
]
for cfg in cfgs:
    beit = BuildBackbone(cfg)
    state_dict = loadpretrainedweights(
        structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict = beit.beitconvert(state_dict)
    state_dict = beit.resizerelposembed(state_dict)
    try:
        beit.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        beit.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)