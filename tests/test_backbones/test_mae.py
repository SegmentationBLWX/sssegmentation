'''
Function:
    Implementation of Testing MAE
Author:
    Zhenchao Jin
'''
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.mae import DEFAULT_MODEL_URLS


'''MAEs'''
cfgs = [{
    'type': 'MAE', 'structure_type': 'mae_pretrain_vit_base', 'pretrained': True, 
    'img_size': (512, 512), 'patch_size': 16, 'embed_dims': 768, 'num_layers': 12,
    'num_heads': 12, 'mlp_ratio': 4, 'init_values': 1.0, 'drop_path_rate': 0.1,
    'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6},
}]
for cfg in cfgs:
    mae = BuildBackbone(cfg)
    state_dict = loadpretrainedweights(
        structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict = mae.beitconvert(state_dict)
    state_dict = mae.resizerelposembed(state_dict)
    try:
        mae.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        mae.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)