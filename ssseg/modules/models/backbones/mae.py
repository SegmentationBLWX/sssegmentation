'''
Function:
    Implementation of MAE
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .beit import BEiT
from .beit import BEiTTransformerEncoderLayer as MAETransformerEncoderLayer


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'mae_pretrain_vit_base': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''MAE'''
class MAE(BEiT):
    def __init__(self, structure_type, img_size=(640, 640), patch_size=16, in_channels=3, embed_dims=768, num_layers=12, num_heads=12,
                 mlp_ratio=4, out_indices=(3, 5, 7, 11), attn_drop_rate=0., drop_path_rate=0.1, norm_cfg={'type': 'LayerNorm', 'eps': 1e-6}, act_cfg={'type': 'GELU'},
                 patch_norm=False, final_norm=False, num_fcs=2, init_values=1.0, pretrained=True, pretrained_model_path=''):
        super(MAE, self).__init__(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, out_indices=out_indices, qv_bias=False, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg, act_cfg=act_cfg, patch_norm=patch_norm, final_norm=final_norm, num_fcs=num_fcs, init_values=init_values, 
            pretrained=False, pretrained_model_path=pretrained_model_path, structure_type=structure_type
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dims))
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''buildlayers'''
    def buildlayers(self):
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(MAETransformerEncoderLayer(
                embed_dims=self.embed_dims, num_heads=self.num_heads, feedforward_channels=self.mlp_ratio * self.embed_dims, attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=dpr[i], num_fcs=self.num_fcs, bias=True, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg, window_size=self.patch_shape, init_values=self.init_values
            ))
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type='mae_pretrain_vit_base', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.beitconvert(state_dict)
        state_dict = self.resizerelposembed(state_dict)
        state_dict = self.resizeabsposembed(state_dict)
        self.load_state_dict(state_dict, strict=False)
    '''resizeabsposembed'''
    def resizeabsposembed(self, state_dict):
        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = self.pos_embed.shape[-2] - self.num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
            # height (== width) for the new position embedding
            new_size = int(self.num_patches**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                state_dict['pos_embed'] = new_pos_embed
        return state_dict
    '''forward'''
    def forward(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        outs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if idx in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)