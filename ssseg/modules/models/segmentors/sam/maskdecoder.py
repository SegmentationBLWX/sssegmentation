'''
Function:
    Implementation of MaskDecoder
Author:
    Zhenchao Jin
'''
import torch
from torch import nn
from torch.nn import functional as F
from ...backbones import BuildActivation
from .transformer import TwoWayTransformer
from ...backbones.samvit import LayerNorm2d


'''MLP'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
    '''forward'''
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


'''MaskDecoder'''
class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim, transformer_cfg, num_multimask_outputs=3, act_cfg={'type': 'GELU'}, iou_head_depth=3, iou_head_hidden_dim=256):
        super(MaskDecoder, self).__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(**transformer_cfg)
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            BuildActivation(act_cfg=act_cfg),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            BuildActivation(act_cfg=act_cfg),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)
    '''forward'''
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        masks, iou_pred = self.predictmasks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # prepare output
        return masks, iou_pred
    '''predictmasks'''
    def predictmasks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings):
        # concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        # upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        # prepare output
        return masks, iou_pred