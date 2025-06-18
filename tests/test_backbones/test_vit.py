'''
Function:
    Implementation of Testing ViT
Author:
    Zhenchao Jin
'''
import math
from ssseg.modules import BuildBackbone, loadpretrainedweights
from ssseg.modules.models.backbones.vit import DEFAULT_MODEL_URLS


'''ViTs'''
cfgs = [
{'type': 'VisionTransformer', 'structure_type': 'jx_vit_large_p16_384', 'img_size': (512, 512), 'out_indices': (9, 14, 19, 23),
 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6}, 'pretrained': True, 'selected_indices': (0, 1, 2, 3),
 'patch_size': 16, 'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4,
 'qkv_bias': True, 'drop_rate': 0.1, 'attn_drop_rate': 0., 'drop_path_rate': 0., 'with_cls_token': True,
 'output_cls_token': False, 'patch_norm': False, 'final_norm': False, 'num_fcs': 2,}
]
for cfg in cfgs:
    vit = BuildBackbone(backbone_cfg=cfg)
    state_dict = loadpretrainedweights(
        structure_type=cfg['structure_type'], pretrained_model_path='', default_model_urls=DEFAULT_MODEL_URLS
    )
    state_dict = vit.vitconvert(state_dict)
    if 'pos_embed' in state_dict.keys():
        if vit.pos_embed.shape != state_dict['pos_embed'].shape:
            h, w = vit.img_size
            pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1] - 1))
            state_dict['pos_embed'] = vit.resizeposembed(state_dict['pos_embed'], (h // vit.patch_size, w // vit.patch_size), (pos_size, pos_size), vit.interpolate_mode)
    try:
        vit.load_state_dict(state_dict, strict=False)
    except Exception as err:
        print(err)
    try:
        vit.load_state_dict(state_dict, strict=True)
    except Exception as err:
        print(err)