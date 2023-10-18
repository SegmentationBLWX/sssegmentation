'''
Function:
    Implementation of MSDeformAttnPixelDecoder
Author:
    Zhenchao Jin
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .misc import getclones
from ....backbones import PositionEmbeddingSine, BuildActivation, BuildNormalization
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as MSDeformAttnFunction
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch as ms_deform_attn_core_pytorch
except:
    MSDeformAttnFunction = None
    ms_deform_attn_core_pytorch = None


'''MSDeformAttn'''
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super(MSDeformAttn, self).__init__()
        # assert
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads)
        # set attributes
        self.im2col_step = 128
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        # define layers
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # reset parameters
        self.resetparameters()
    '''resetparameters'''
    def resetparameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    '''forward'''
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        # assert
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        # feed to value project
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # sampling_offsets
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # attention_weights
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # MSDeformAttnFunction
        try:
            output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        except:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        # return output
        return output


'''MSDeformAttnTransformerEncoderLayer'''
class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, act_cfg={'type': 'ReLU', 'inplace': True}, n_levels=4, n_heads=8, n_points=4):
        super(MSDeformAttnTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = BuildActivation(act_cfg)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    '''withposembed'''
    @staticmethod
    def withposembed(tensor, pos):
        return tensor if pos is None else tensor + pos
    '''forwardffn'''
    def forwardffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    '''forward'''
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.withposembed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forwardffn(src)
        # return
        return src


'''MSDeformAttnTransformerEncoder'''
class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(MSDeformAttnTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = getclones(encoder_layer, num_layers)
    '''getreferencepoints'''
    @staticmethod
    def getreferencepoints(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    '''forward'''
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.getreferencepoints(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


'''MSDeformAttnTransformerEncoderOnly'''
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, act_cfg={'type': 'ReLU', 'inplace': True}, num_feature_levels=4, enc_n_points=4):
        super(MSDeformAttnTransformerEncoderOnly, self).__init__()
        # set attributes
        self.nhead = nhead
        self.d_model = d_model
        # define layers
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, act_cfg, num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        # reset parameters
        self.resetparameters()
    '''resetparameters'''
    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m.resetparameters()
        nn.init.normal_(self.level_embed)
    '''getvalidratio'''
    def getvalidratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    '''forward'''
    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.getvalidratio(m) for m in masks], 1)
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # return
        return memory, spatial_shapes, level_start_index


'''MSDeformAttnPixelDecoder'''
class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, input_shape, transformer_dropout, transformer_nheads, transformer_dim_feedforward, transformer_enc_layers, conv_dim, mask_dim, norm_cfg, act_cfg, transformer_in_features, common_stride):
        super(MSDeformAttnPixelDecoder, self).__init__()
        import fvcore.nn.weight_init as weight_init
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        # to decide extra FPN layers
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim),
            )])
        # initialize input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # define transformer layers
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, apply_normalize=True)
        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.mask_features)
        # always use 3 scales
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs, output_convs = nn.ModuleList(), nn.ModuleList()
        use_bias = (norm_cfg is None)
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, conv_dim, kernel_size=1, stride=1, padding=0, bias=use_bias),
                BuildNormalization(placeholder=conv_dim, norm_cfg=norm_cfg),
            )
            output_conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                BuildNormalization(placeholder=conv_dim, norm_cfg=norm_cfg),
                BuildActivation(act_cfg=act_cfg),
            )
            weight_init.c2_xavier_fill(lateral_conv[0])
            weight_init.c2_xavier_fill(output_conv[0])
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
    '''forwardfeatures'''
    def forwardfeatures(self, features):
        srcs, pos = [], []
        # reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            # deformable detr does not support half precision
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out, multi_scale_features, num_cur_levels = [], [], 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        # append `out` with extra FPN levels, reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv, output_conv = self.lateral_convs[idx], self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)
        # return outputs
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features