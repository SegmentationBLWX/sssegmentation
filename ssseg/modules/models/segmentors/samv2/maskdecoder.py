'''
Function:
    Implementation of MaskDecoder
Author:
    Zhenchao Jin
'''
import torch
from torch import nn
from ...backbones.hiera import MLP
from ...backbones import BuildActivation
from ...backbones.samvit import LayerNorm2d


'''MaskDecoder'''
class MaskDecoder(nn.Module):
    def __init__(self, *, transformer_dim, transformer, num_multimask_outputs=3, act_cfg={'type': 'GELU'}, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=False,
                 iou_prediction_use_sigmoid=False, dynamic_multimask_via_stability=False, dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98, pred_obj_scores=False,
                 pred_obj_scores_mlp=False, use_multimask_token_for_obj_ptr=False):
        super(MaskDecoder, self).__init__()
        # set attributes and embeddings
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        # output upscaling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4), BuildActivation(act_cfg=act_cfg),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), BuildActivation(act_cfg=act_cfg),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)
        # mlps
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, sigmoid_output=iou_prediction_use_sigmoid)
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
        # when outputting a single mask, optionally we can dynamically fall back to the best multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
    '''forward'''
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, repeat_image, high_res_features=None):
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predictmasks(
            image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image, high_res_features=high_res_features,
        )
        # select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self.dynamicmultimaskviastability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        # return
        return masks, iou_pred, sam_tokens_out, object_score_logits
    '''predictmasks'''
    def predictmasks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, repeat_image, high_res_features=None):
        # concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim (from `getdensepe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]
        # upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        # return
        return masks, iou_pred, mask_tokens_out, object_score_logits
    '''getstabilityscores'''
    def getstabilityscores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores
    '''dynamicmultimaskviastability'''
    def dynamicmultimaskviastability(self, all_mask_logits, all_iou_scores):
        # the best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)
        # the mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self.getstabilityscores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh
        # dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(is_stable[..., None, None].expand_as(singlemask_logits), singlemask_logits, best_multimask_logits)
        iou_scores_out = torch.where(is_stable.expand_as(singlemask_iou_scores), singlemask_iou_scores, best_multimask_iou_scores)
        # return
        return mask_logits_out, iou_scores_out