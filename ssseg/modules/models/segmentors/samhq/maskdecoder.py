'''
Function:
    Implementation of MaskDecoder
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ...backbones.samvit import LayerNorm2d
from ..sam.maskdecoder import MaskDecoder, MLP


'''MaskDecoderHQ'''
class MaskDecoderHQ(MaskDecoder):
    def __init__(self, transformer_dim, transformer_cfg, num_multimask_outputs=3, act_cfg={'type': 'GELU'}, iou_head_depth=3, iou_head_hidden_dim=256, vit_dim=1024):
        super(MaskDecoderHQ, self).__init__(
            transformer_dim=transformer_dim, transformer_cfg=transformer_cfg, num_multimask_outputs=num_multimask_outputs, act_cfg=act_cfg, 
            iou_head_depth=iou_head_depth, iou_head_hidden_dim=iou_head_hidden_dim,
        )
        # samhq parameters
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1
        # three conv fusion layers for obtaining HQ features
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1)
        )
    '''forward'''
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, hq_token_only, interm_embeddings):
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        masks, iou_pred = self.predictmasks(
            image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, 
            dense_prompt_embeddings=dense_prompt_embeddings, hq_features=hq_features,
        )
        # select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens-1)
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred, dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            masks_sam = masks[:, mask_slice]
        masks_hq = masks[:, slice(self.num_mask_tokens-1, self.num_mask_tokens)]
        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq
        # prepare output
        return masks, iou_pred
    '''predictmasks'''
    def predictmasks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, hq_features):
        # concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
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
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]
        # upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b, 1, 1, 1)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in[:, :self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:, self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_sam_hq], dim=1)
        # generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        # return result
        return masks, iou_pred