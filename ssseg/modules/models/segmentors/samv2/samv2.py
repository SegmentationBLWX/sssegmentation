'''
Function:
    Implementation of SAMV2
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from PIL.Image import Image
from ..base import BaseSegmentor
from ...backbones.hiera import MLP
from collections import OrderedDict
from .maskdecoder import MaskDecoder
from ...backbones import BuildBackbone
from .transforms import SAMV2Transforms
from .memoryencoder import MemoryEncoder
from .promptencoder import PromptEncoder
from .transformer import TwoWayTransformer
from .memoryattention import MemoryAttention
from torchvision.ops.boxes import batched_nms, box_area
from .misc import selectclosestcondframes, get1dsinepe, loadvideoframes, fillholesinmaskscores, concatpoints
from .amg import (
    buildalllayerpointgrids, areafromrle, batchiterator, batchedmasktobox, boxxyxytoxywh, calculatestabilityscore, cocoencoderle, generatecropboxes, isboxnearcropedge,
    masktorlepytorch, MaskData, removesmallregions, rletomask, uncropboxesxyxy, uncropmasks, uncroppoints
)


'''a large negative value as a placeholder score for missing objects'''
NO_OBJ_SCORE = -1024.0


'''SAMV2'''
class SAMV2(BaseSegmentor):
    def __init__(self, cfg, mode):
        backbone = cfg.pop('backbone')
        super(SAMV2, self).__init__(cfg=cfg, mode=mode)
        cfg['backbone'] = backbone
        assert mode in ['TEST'], f'only support TEST mode for {self.__class__.__name__}'
        # Part 1: the image backbone
        self.image_encoder = BuildBackbone(cfg['backbone'])
        # use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = cfg['head']['use_high_res_features_in_sam']
        self.num_feature_levels = 3 if cfg['head']['use_high_res_features_in_sam'] else 1
        self.use_obj_ptrs_in_encoder = cfg['head']['use_obj_ptrs_in_encoder']
        self.max_obj_ptrs_in_encoder = cfg['head']['max_obj_ptrs_in_encoder']
        if cfg['head']['use_obj_ptrs_in_encoder']:
            self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = cfg['head']['add_tpos_enc_to_obj_ptrs']
        if cfg['head']['proj_tpos_enc_in_obj_ptrs']:
            assert cfg['head']['add_tpos_enc_to_obj_ptrs']
        self.proj_tpos_enc_in_obj_ptrs = cfg['head']['proj_tpos_enc_in_obj_ptrs']
        self.only_obj_ptrs_in_the_past_for_eval = cfg['head']['only_obj_ptrs_in_the_past_for_eval']
        # Part 2: memory attention to condition current frame's visual features with memories (and obj ptrs) from past frames
        self.memory_attention = MemoryAttention(**cfg['head']['memory_attention_cfg'])
        self.hidden_dim = self.memory_attention.d_model
        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = MemoryEncoder(**cfg['head']['memory_encoder_cfg'])
        self.mem_dim = self.hidden_dim
        # if there is compression of memories along channel dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        # number of memories accessible
        self.num_maskmem = cfg['head']['num_maskmem']
        # temporal encoding of the memories
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(cfg['head']['num_maskmem'], 1, 1, self.mem_dim))
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.no_mem_embed, std=0.02)
        nn.init.trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = cfg['head']['directly_add_no_mem_embed']
        # apply sigmoid to the output raw mask logits (to turn them from range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = cfg['head']['sigmoid_scale_for_mem_enc']
        self.sigmoid_bias_for_mem_enc = cfg['head']['sigmoid_bias_for_mem_enc']
        self.binarize_mask_from_pts_for_mem_enc = cfg['head']['binarize_mask_from_pts_for_mem_enc']
        self.non_overlap_masks_for_mem_enc = cfg['head']['non_overlap_masks_for_mem_enc']
        self.memory_temporal_stride_for_eval = cfg['head']['memory_temporal_stride_for_eval']
        # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = cfg['head']['use_mask_input_as_output_without_sam']
        self.multimask_output_in_sam = cfg['head']['multimask_output_in_sam']
        self.multimask_min_pt_num = cfg['head']['multimask_min_pt_num']
        self.multimask_max_pt_num = cfg['head']['multimask_max_pt_num']
        self.multimask_output_for_tracking = cfg['head']['multimask_output_for_tracking']
        self.use_multimask_token_for_obj_ptr = cfg['head']['use_multimask_token_for_obj_ptr']
        self.iou_prediction_use_sigmoid = cfg['head']['iou_prediction_use_sigmoid']
        # Part 4: SAM-style prompt encoder (for both mask and point inputs) and SAM-style mask decoder for the final mask output
        self.image_size = cfg['head']['image_size']
        self.backbone_stride = cfg['head']['backbone_stride']
        self.sam_mask_decoder_extra_args = cfg['head']['sam_mask_decoder_extra_args']
        self.pred_obj_scores = cfg['head']['pred_obj_scores']
        self.pred_obj_scores_mlp = cfg['head']['pred_obj_scores_mlp']
        self.fixed_no_obj_ptr = cfg['head']['fixed_no_obj_ptr']
        self.soft_no_obj_ptr = cfg['head']['soft_no_obj_ptr']
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = nn.Parameter(torch.zeros(1, self.hidden_dim))
            nn.init.trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = cfg['head']['use_mlp_for_obj_ptr_proj']
        self.buildsamheads()
        self.add_all_frames_to_correct_as_cond = cfg['head']['add_all_frames_to_correct_as_cond']
        self.max_cond_frames_in_attn = cfg['head']['max_cond_frames_in_attn']
        # Model compilation
        if cfg['head']['compile_image_encoder']:
            print("Image encoder compilation is enabled. First forward pass will be slow.")
            self.image_encoder.forward = torch.compile(self.image_encoder.forward, mode="max-autotune", fullgraph=True, dynamic=False)
    '''device'''
    @property
    def device(self):
        return next(self.parameters()).device
    '''forward'''
    def forward(self, data_meta):
        raise NotImplementedError(f'train {self.__class__.__name__} not to be implemented')
    '''buildsamheads'''
    def buildsamheads(self):
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        # prompt encoder
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim, image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size), 
            input_image_size=(self.image_size, self.image_size), mask_in_chans=16,
        )
        # mask decoder
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=self.sam_prompt_embed_dim, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid, pred_obj_scores=self.pred_obj_scores, pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr, **(self.sam_mask_decoder_extra_args or {}),
        )
        # use_obj_ptrs_in_encoder
        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = nn.Identity()
        # proj_tpos_enc_in_obj_ptrs
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = nn.Identity()
    '''forwardsamheads'''
    def forwardsamheads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        # prepare and assert
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size
        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        # b) Handle mask prompts
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(mask_inputs.float(), size=self.sam_prompt_encoder.mask_input_size, align_corners=False, mode="bilinear", antialias=True)
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt)
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features, image_pe=self.sam_prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, repeat_image=False, high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)
        # convert masks from possibly bfloat16 (or float16) to float32 (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        sam_output_token = sam_output_tokens[:, 0]
        # multimask_output
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        # extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        # return
        return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits
    '''usemaskasoutput'''
    def usemaskasoutput(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(high_res_masks, size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4), align_corners=False, mode="bilinear", antialias=True)
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = torch.zeros(mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device)
        else:
            _, _, _, _, _, obj_ptr, _ = self.forwardsamheads(backbone_features=backbone_features, mask_inputs=self.mask_downsample(mask_inputs_float), high_res_features=high_res_features)
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits
    '''forwardimage'''
    def forwardimage(self, img_batch):
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out
    '''preparebackbonefeatures'''
    def preparebackbonefeatures(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
    '''preparememoryconditionedfeatures'''
    def preparememoryconditionedfeatures(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, output_dict, num_frames, track_in_reverse=False):
        # basic information
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        # the case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images. in this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = selectclosestcondframes(frame_idx, cond_outputs, self.max_cond_frames_in_attn)
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].cuda(non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {t: out for t, out in selected_cond_outputs.items() if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [(abs(frame_idx - t), out["obj_ptr"]) for t, out in ptr_cond_outputs.items()]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get1dsinepe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.memory_attention(curr=current_vision_feats, curr_pos=current_vision_pos_embeds, memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=num_obj_ptr_tokens)
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        # return
        return pix_feat_with_mem
    '''encodenewmemory'''
    def encodenewmemory(self, current_vision_feats, feat_sizes, pred_masks_high_res, is_mask_from_pts):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self.applynonoverlappingconstraints(pred_masks_high_res)
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        return maskmem_features, maskmem_pos_enc
    '''trackstep'''
    def trackstep(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                  output_dict, num_frames, track_in_reverse=False, run_mem_encoder=True, prev_sam_mask_logits=None):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self.usemaskasoutput(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat_with_mem = self.preparememoryconditionedfeatures(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame, current_vision_feats=current_vision_feats[-1:], current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:], output_dict=output_dict, num_frames=num_frames, track_in_reverse=track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self.usemultimask(is_init_cond_frame, point_inputs)
            sam_outputs = self.forwardsamheads(backbone_features=pix_feat_with_mem, point_inputs=point_inputs, mask_inputs=mask_inputs, high_res_features=high_res_features, multimask_output=multimask_output)
        _, _, _, low_res_masks, high_res_masks, obj_ptr, _ = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self.encodenewmemory(
                current_vision_feats=current_vision_feats, feat_sizes=feat_sizes, pred_masks_high_res=high_res_masks_for_mem_enc, is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None
        return current_out
    '''usemultimask'''
    def usemultimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = self.multimask_output_in_sam and (is_init_cond_frame or self.multimask_output_for_tracking) and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        return multimask_output
    '''applynonoverlappingconstraints'''
    def applynonoverlappingconstraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks
        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


'''SAMV2ImagePredictor'''
class SAMV2ImagePredictor(nn.Module):
    def __init__(self, samv2_cfg=None, use_default_samv2_t=False, use_default_samv2_s=False, use_default_samv2_bplus=False, use_default_samv2_l=False,
                 device='cuda', load_ckpt_strict=True, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0, apply_postprocessing=True):
        super(SAMV2ImagePredictor, self).__init__()
        # build sam model
        if samv2_cfg is None:
            samv2_cfg = {
                'backbone': {
                    'type': 'HieraWithFPN', 'scalp': 1.0,
                    'hiera_cfg': {
                        'embed_dim': 144, 'num_heads': 2, 'stages': [2, 6, 36, 4], 'global_att_blocks': [23, 33, 43], 'window_pos_embed_bkg_spatial_size': [7, 7], 'window_spec': [8, 4, 16, 8],
                    }, 
                    'fpn_cfg': {
                        'd_model': 256, 'backbone_channel_list': [1152, 576, 288, 144], 'fpn_top_down_levels': [2, 3], 'fpn_interp_model': 'nearest',
                        'position_encoding_cfg': dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'),
                    },
                },
                'head': {
                    'memory_attention_cfg': {
                        'd_model': 256, 'pos_enc_at_input': True, 'num_layers': 4, 
                        'layer_cfg': {
                            'act_cfg': {'type': 'ReLU'}, 'dim_feedforward': 2048, 'dropout': 0.1, 'pos_enc_at_attn': False, 'd_model': 256, 'pos_enc_at_cross_attn_keys': True, 'pos_enc_at_cross_attn_queries': False,
                            'self_attention_cfg': dict(type='RoPEAttention', rope_theta=10000.0, feat_sizes=[32, 32], embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1),
                            'cross_attention_cfg': dict(type='RoPEAttention', rope_theta=10000.0, feat_sizes=[32, 32], rope_k_repeat=True, embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1, kv_in_dim=64),
                        }
                    },
                    'memory_encoder_cfg': {
                        'out_dim': 64, 'position_encoding_cfg': dict(num_pos_feats=64, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'),
                        'mask_downsampler_cfg': dict(kernel_size=3, stride=2, padding=1), 'fuser_cfg': dict(num_layers=2, layer_cfg=dict(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True)),
                    },
                    'num_maskmem': 7, 'image_size': 1024, 'backbone_stride': 16, 'sigmoid_scale_for_mem_enc': 20.0, 'sigmoid_bias_for_mem_enc': -10.0, 'binarize_mask_from_pts_for_mem_enc': False,
                    'use_mask_input_as_output_without_sam': True, 'max_cond_frames_in_attn': -1, 'directly_add_no_mem_embed': True, 'use_high_res_features_in_sam': True, 'multimask_output_in_sam': True,
                    'multimask_min_pt_num': 0, 'multimask_max_pt_num': 1, 'multimask_output_for_tracking': True, 'use_multimask_token_for_obj_ptr': True, 'iou_prediction_use_sigmoid': True,
                    'memory_temporal_stride_for_eval': 1, 'add_all_frames_to_correct_as_cond': False, 'non_overlap_masks_for_mem_enc': False, 'use_obj_ptrs_in_encoder': True, 'max_obj_ptrs_in_encoder': 16,
                    'add_tpos_enc_to_obj_ptrs': False, 'proj_tpos_enc_in_obj_ptrs': False, 'only_obj_ptrs_in_the_past_for_eval': True, 'pred_obj_scores': True, 'pred_obj_scores_mlp': True, 'fixed_no_obj_ptr': True,
                    'soft_no_obj_ptr': False, 'use_mlp_for_obj_ptr_proj': True, 'sam_mask_decoder_extra_args': None, 'compile_image_encoder': False,
                },
            }
            if use_default_samv2_l:
                assert (not use_default_samv2_t) and (not use_default_samv2_s) and (not use_default_samv2_bplus)
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
            elif use_default_samv2_bplus:
                assert (not use_default_samv2_t) and (not use_default_samv2_s) and (not use_default_samv2_l)
                samv2_cfg['backbone']['hiera_cfg'] = dict(embed_dim=112, num_heads=2)
                samv2_cfg['backbone']['fpn_cfg'] = dict(
                    position_encoding_cfg=dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), 
                    d_model=256, backbone_channel_list=[896, 448, 224, 112], fpn_top_down_levels=[2, 3], fpn_interp_model='nearest',
                )
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt'
            elif use_default_samv2_s:
                assert (not use_default_samv2_t) and (not use_default_samv2_bplus) and (not use_default_samv2_l)
                samv2_cfg['backbone']['hiera_cfg'] = dict(embed_dim=96, num_heads=1, stages=[1, 2, 11, 2], global_att_blocks=[7, 10, 13], window_pos_embed_bkg_spatial_size=[7, 7])
                samv2_cfg['backbone']['fpn_cfg'] = dict(
                    position_encoding_cfg=dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), 
                    d_model=256, backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2, 3], fpn_interp_model='nearest',
                )
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt'
            elif use_default_samv2_t:
                assert (not use_default_samv2_s) and (not use_default_samv2_bplus) and (not use_default_samv2_l)
                samv2_cfg['backbone']['hiera_cfg'] = dict(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7])
                samv2_cfg['backbone']['fpn_cfg'] = dict(
                    position_encoding_cfg=dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), 
                    d_model=256, backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2, 3], fpn_interp_model='nearest',
                )
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt'
        else:
            assert (not use_default_samv2_t) and (not use_default_samv2_s) and (not use_default_samv2_bplus) and (not use_default_samv2_l)
        self.model = self.buildsamv2(samv2_cfg=samv2_cfg, device=device, apply_postprocessing=apply_postprocessing)
        if 'ckptpath' in samv2_cfg and (os.path.exists(samv2_cfg['ckptpath']) or samv2_cfg['ckptpath'].startswith('https')):
            if os.path.exists(samv2_cfg['ckptpath']):
                with open(samv2_cfg['ckptpath'], 'rb') as fp:
                    state_dict = torch.load(fp, map_location='cpu')
            elif samv2_cfg['ckptpath'].startswith('https'):
                state_dict = model_zoo.load_url(samv2_cfg['ckptpath'], map_location='cpu')
            else:
                raise ValueError('ckptpath %s could not be loaded.' % samv2_cfg['ckptpath'])
            self.model.load_state_dict(state_dict['model'], strict=load_ckpt_strict)
        # build transforms
        self._transforms = SAMV2Transforms(resolution=self.model.image_size, mask_threshold=mask_threshold, max_hole_area=max_hole_area, max_sprinkle_area=max_sprinkle_area)
        # predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # whether the predictor is set for single image or a batch of images
        self._is_batch = False
        # predictor config
        self.mask_threshold = mask_threshold
        # spatial dim for backbone feature maps
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
    '''buildsamv2'''
    def buildsamv2(self, samv2_cfg, device, apply_postprocessing=True):
        if apply_postprocessing:
            samv2_cfg['head']['sam_mask_decoder_extra_args'] = {
                'dynamic_multimask_via_stability': True,
                'dynamic_multimask_stability_delta': 0.05,
                'dynamic_multimask_stability_thresh': 0.98,
            }
        samv2_model = SAMV2(cfg=samv2_cfg, mode='TEST')
        samv2_model.to(device=device)
        samv2_model.eval()
        return samv2_model
    '''setimage'''
    @torch.no_grad()
    def setimage(self, image):
        self.resetpredictor()
        # transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            assert image.shape[-1] <= 3, 'For numpy array image, we assume (HxWxC) format'
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported.")
        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)
        assert len(input_image.shape) == 4 and input_image.shape[1] == 3 and input_image.shape[0] == 1, f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        # computing image embeddings for the provided image
        backbone_out = self.model.forwardimage(input_image)
        _, vision_feats, _, _ = self.model.preparebackbonefeatures(backbone_out)
        # add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        # construct results
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
    '''setimagebatch'''
    @torch.no_grad()
    def setimagebatch(self, image_list):
        # initialize
        self.resetpredictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(image, np.ndarray) and (image.shape[-1] <= 3), "images are expected to be an np.ndarray in RGB format, and of shape HxWxC"
            self._orig_hw.append(image.shape[:2])
        # transform the image to the form expected by the model
        img_batch = self._transforms.forwardbatch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert len(img_batch.shape) == 4 and img_batch.shape[1] == 3, f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        # computing image embeddings for the provided images
        backbone_out = self.model.forwardimage(img_batch)
        _, vision_feats, _, _ = self.model.preparebackbonefeatures(backbone_out)
        # add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        # construct results
        feats = [feat.permute(1, 2, 0).view(batch_size, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
    '''predictbatch'''
    def predictbatch(self, point_coords_batch=None, point_labels_batch=None, box_batch=None, mask_input_batch=None, multimask_output=True, return_logits=False, normalize_coords=True):
        # assert
        assert self._is_batch, "this function should only be used when in batched mode"
        assert self._is_image_set, "an image must be set with .setimagebatch(...) before mask prediction."
        # iter to predict
        num_images = len(self._features["image_embed"])
        all_masks, all_ious, all_low_res_masks = [], [], []
        for img_idx in range(num_images):
            point_coords = point_coords_batch[img_idx] if point_coords_batch is not None else None
            point_labels = point_labels_batch[img_idx] if point_labels_batch is not None else None
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = mask_input_batch[img_idx] if mask_input_batch is not None else None
            mask_input, unnorm_coords, labels, unnorm_box = self.prepprompts(point_coords, point_labels, box, mask_input, normalize_coords, img_idx=img_idx)
            masks, iou_predictions, low_res_masks = self.purepredict(unnorm_coords, labels, unnorm_box, mask_input, multimask_output, return_logits=return_logits, img_idx=img_idx)
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)
        # return
        return all_masks, all_ious, all_low_res_masks
    '''predict'''
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False, normalize_coords=True):
        assert self._is_image_set, "an image must be set with .setimage(...) before mask prediction."
        # transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self.prepprompts(point_coords, point_labels, box, mask_input, normalize_coords)
        # predict
        masks, iou_predictions, low_res_masks = self.purepredict(unnorm_coords, labels, unnorm_box, mask_input, multimask_output, return_logits=return_logits)
        # convert
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        # return
        return masks_np, iou_predictions_np, low_res_masks_np
    '''prepprompts'''
    def prepprompts(self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        # point_coords
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            unnorm_coords = self._transforms.transformcoords(point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx])
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        # box
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transformboxes(box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx])
        # mask_logits
        if mask_logits is not None:
            mask_input = torch.as_tensor(mask_logits, dtype=torch.float, device=self.device)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        # return
        return mask_input, unnorm_coords, labels, unnorm_box
    '''purepredict'''
    @torch.no_grad()
    def purepredict(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output=True, return_logits=False, img_idx=-1):
        assert self._is_image_set, "an image must be set with .setimage(...) before mask prediction."
        # concat_points
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None
        # embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(points=concat_points, boxes=None, masks=mask_input)
        # predict masks
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in self._features["high_res_feats"]]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0), image_pe=self.model.sam_prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, repeat_image=batched_mode, high_res_features=high_res_features,
        )
        # upscale the masks to the original image resolution
        masks = self._transforms.postprocessmasks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        # return
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks
    '''getimageembedding'''
    def getimageembedding(self):
        assert self._is_image_set, "an image must be set with .setimage(...) to generate an embedding."
        assert self._features is not None, "features must exist if an image has been set."
        return self._features["image_embed"]
    '''device'''
    @property
    def device(self):
        return self.model.device
    '''resetpredictor'''
    def resetpredictor(self):
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False


'''SAMV2AutomaticMaskGenerator'''
class SAMV2AutomaticMaskGenerator(nn.Module):
    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8, stability_score_thresh=0.95, stability_score_offset=1.0, mask_threshold=0.0, box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, 
                 crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0, output_mode="binary_mask", use_m2m=False, multimask_output=True, samv2_cfg=None, use_default_samv2_t=False,
                 use_default_samv2_s=False, use_default_samv2_bplus=False, use_default_samv2_l=True, device='cuda', load_ckpt_strict=True, apply_postprocessing=False):
        super(SAMV2AutomaticMaskGenerator, self).__init__()
        # deal with points_per_side and point_grids
        assert (points_per_side is None) != (point_grids is None), "exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = buildalllayerpointgrids(points_per_side, crop_n_layers, crop_n_points_downscale_factor)
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")
        # output_mode
        assert output_mode in ["binary_mask", "uncompressed_rle", "coco_rle"], f"unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils
            except ImportError as e:
                print("please install pycocotools")
                raise e
        # predictor
        self.predictor = SAMV2ImagePredictor(
            samv2_cfg=samv2_cfg, use_default_samv2_l=use_default_samv2_l, use_default_samv2_bplus=use_default_samv2_bplus, use_default_samv2_s=use_default_samv2_s, use_default_samv2_t=use_default_samv2_t, 
            device=device, load_ckpt_strict=load_ckpt_strict, max_hole_area=min_mask_region_area, max_sprinkle_area=min_mask_region_area, apply_postprocessing=apply_postprocessing,
        )
        # set attributes
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output
    '''generate'''
    @torch.no_grad()
    def generate(self, image):
        # generate masks
        mask_data = self.generatemasks(image)
        # encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [cocoencoderle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rletomask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]
        # write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx], "area": areafromrle(mask_data["rles"][idx]), "bbox": boxxyxytoxywh(mask_data["boxes"][idx]).tolist(), "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()], "stability_score": mask_data["stability_score"][idx].item(), "crop_box": boxxyxytoxywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)
        # return
        return curr_anns
    '''generatemasks'''
    def generatemasks(self, image):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generatecropboxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
        # iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self.processcrop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)
        # remove duplicate masks between crops
        if len(crop_boxes) > 1:
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(data["boxes"].float(), scores, torch.zeros_like(data["boxes"][:, 0]), iou_threshold=self.crop_nms_thresh)
            data.filter(keep_by_nms)
        data.tonumpy()
        # return
        return data
    '''processcrop'''
    def processcrop(self, image, crop_box, crop_layer_idx, orig_size):
        # crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.setimage(cropped_im)
        # get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale
        # generate masks for this crop in batches
        data = MaskData()
        for (points,) in batchiterator(self.points_per_batch, points_for_image):
            batch_data = self.processbatch(points, cropped_im_size, crop_box, orig_size, normalize=True)
            data.cat(batch_data)
            del batch_data
        self.predictor.resetpredictor()
        # remove duplicates within this crop.
        keep_by_nms = batched_nms(data["boxes"].float(), data["iou_preds"], torch.zeros_like(data["boxes"][:, 0]), iou_threshold=self.box_nms_thresh)
        data.filter(keep_by_nms)
        # return to the original image frame
        data["boxes"] = uncropboxesxyxy(data["boxes"], crop_box)
        data["points"] = uncroppoints(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        # return
        return data
    '''processbatch'''
    def processbatch(self, points, im_size, crop_box, orig_size, normalize=False):
        orig_h, orig_w = orig_size
        # run model on this batch
        points = torch.as_tensor(points, device=self.predictor.device)
        in_points = self.predictor._transforms.transformcoords(points, normalize=normalize, orig_hw=im_size)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, low_res_masks = self.predictor.purepredict(in_points[:, None, :], in_labels[:, None], multimask_output=self.multimask_output, return_logits=True)
        # serialize predictions and store in MaskData
        data = MaskData(masks=masks.flatten(0, 1), iou_preds=iou_preds.flatten(0, 1), points=points.repeat_interleave(masks.shape[1], dim=0), low_res_masks=low_res_masks.flatten(0, 1))
        del masks
        # start
        if not self.use_m2m:
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)
            data["stability_score"] = calculatestabilityscore(data["masks"], self.mask_threshold, self.stability_score_offset)
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            in_points = self.predictor._transforms.transformcoords(data["points"], normalize=normalize, orig_hw=im_size)
            labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            masks, ious = self.refinewithm2m(in_points, labels, data["low_res_masks"], self.points_per_batch)
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)
            data["stability_score"] = calculatestabilityscore(data["masks"], self.mask_threshold, self.stability_score_offset)
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        # threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batchedmasktobox(data["masks"])
        # filter boxes that touch crop boundaries
        keep_mask = ~isboxnearcropedge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        # compress to RLE
        data["masks"] = uncropmasks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = masktorlepytorch(data["masks"])
        del data["masks"]
        # return
        return data
    '''postprocesssmallregions'''
    @staticmethod
    def postprocesssmallregions(mask_data, min_area, nms_thresh):
        if len(mask_data["rles"]) == 0:
            return mask_data
        # filter small disconnected regions and holes
        new_masks, scores = [], []
        for rle in mask_data["rles"]:
            mask = rletomask(rle)
            mask, changed = removesmallregions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = removesmallregions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(float(unchanged))
        # recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batchedmasktobox(masks)
        keep_by_nms = batched_nms(boxes.float(), torch.as_tensor(scores), torch.zeros_like(boxes[:, 0]), iou_threshold=nms_thresh)
        # only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = masktorlepytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]
        mask_data.filter(keep_by_nms)
        # return
        return mask_data
    '''refinewithm2m'''
    def refinewithm2m(self, points, point_labels, low_res_masks, points_per_batch):
        new_masks = []
        new_iou_preds = []
        for cur_points, cur_point_labels, low_res_mask in batchiterator(points_per_batch, points, point_labels, low_res_masks):
            best_masks, best_iou_preds, _ = self.predictor.purepredict(
                cur_points[:, None, :], cur_point_labels[:, None], mask_input=low_res_mask[:, None, :], multimask_output=False, return_logits=True,
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
        masks = torch.cat(new_masks, dim=0)
        return masks, torch.cat(new_iou_preds, dim=0)


'''SAMV2VideoPredictor'''
class SAMV2VideoPredictor(SAMV2ImagePredictor):
    def __init__(self, fill_hole_area=0, non_overlap_masks=False, clear_non_cond_mem_around_input=False, clear_non_cond_mem_for_multi_obj=False, **kwargs):
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        super(SAMV2VideoPredictor, self).__init__(**kwargs)
    '''initstate'''
    @torch.inference_mode()
    def initstate(self, video_path, offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=False):
        images, video_height, video_width = loadvideoframes(video_path=video_path, image_size=self.model.image_size, offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames)
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory turning on this option saves the GPU memory at the cost of a lower tracking fps (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = self.device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # a storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        # slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # a temporary storage to hold new outputs when user interact with a frame to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # frames that already holds consolidated outputs from click or mask inputs (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # warm up the visual backbone and cache the image feature on frame 0
        self.getimagefeature(inference_state, frame_idx=0, batch_size=1)
        # return
        return inference_state
    '''buildsamv2'''
    def buildsamv2(self, samv2_cfg, device, apply_postprocessing=True):
        if apply_postprocessing:
            samv2_cfg['head']['sam_mask_decoder_extra_args'] = {
                'dynamic_multimask_via_stability': True,
                'dynamic_multimask_stability_delta': 0.05,
                'dynamic_multimask_stability_thresh': 0.98,
            }
            samv2_cfg['head']['binarize_mask_from_pts_for_mem_enc'] = True
            self.fill_hole_area = 8
        samv2_model = SAMV2(cfg=samv2_cfg, mode='TEST')
        samv2_model.to(device=device)
        samv2_model.eval()
        return samv2_model
    '''objidtoidx'''
    def objidtoidx(self, inference_state, obj_id):
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            inference_state["temp_output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            return obj_idx
        else:
            raise RuntimeError(f"Cannot add new object id {obj_id} after tracking starts. All existing object ids: {inference_state['obj_ids']}. Please call 'resetstate' to restart from scratch.")
    '''objidxtoid'''
    def objidxtoid(self, inference_state, obj_idx):
        return inference_state["obj_idx_to_id"][obj_idx]
    '''getobjnum'''
    def getobjnum(self, inference_state):
        return len(inference_state["obj_idx_to_id"])
    '''addnewpoints'''
    @torch.inference_mode()
    def addnewpoints(self, inference_state, frame_idx, obj_id, points, labels, clear_old_points=True, normalize_coords=True):
        obj_idx = self.objidtoidx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        points = points * self.model.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])
        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concatpoints(point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self.runsingleframeinference(
            inference_state=inference_state, output_dict=obj_output_dict, frame_idx=frame_idx, batch_size=1, is_init_cond_frame=is_init_cond_frame, point_inputs=point_inputs, mask_inputs=None, reverse=reverse, run_mem_encoder=False, prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self.consolidatetempoutputacrossobj(inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True)
        _, video_res_masks = self.getorigvideoresoutput(inference_state, consolidated_out["pred_masks_video_res"])
        return frame_idx, obj_ids, video_res_masks
    '''addnewmask'''
    @torch.inference_mode()
    def addnewmask(self, inference_state, frame_idx, obj_id, mask):
        obj_idx = self.objidtoidx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])
        if mask_H != self.model.image_size or mask_W != self.model.image_size:
            mask_inputs = F.interpolate(mask_inputs_orig, size=(self.model.image_size, self.model.image_size), align_corners=False, mode="bilinear", antialias=True)
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig
        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        current_out, _ = self.runsingleframeinference(
            inference_state=inference_state, output_dict=obj_output_dict, frame_idx=frame_idx, batch_size=1, is_init_cond_frame=is_init_cond_frame, point_inputs=None, mask_inputs=mask_inputs, reverse=reverse, run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self.consolidatetempoutputacrossobj(inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True)
        _, video_res_masks = self.getorigvideoresoutput(inference_state, consolidated_out["pred_masks_video_res"])
        return frame_idx, obj_ids, video_res_masks
    '''getorigvideoresoutput'''
    def getorigvideoresoutput(self, inference_state, any_res_masks):
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = F.interpolate(any_res_masks, size=(video_H, video_W), mode="bilinear", align_corners=False)
        if self.non_overlap_masks:
            video_res_masks = self.model.applynonoverlappingconstraints(video_res_masks)
        return any_res_masks, video_res_masks
    '''consolidatetempoutputacrossobj'''
    def consolidatetempoutputacrossobj(self, inference_state, frame_idx, is_cond, run_mem_encoder, consolidate_at_video_res=False):
        batch_size = self.getobjnum(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.model.image_size // 4
            consolidated_mask_key = "pred_masks"
        consolidated_out = {
            "maskmem_features": None, "maskmem_pos_enc": None, "obj_ptr": torch.full(size=(batch_size, self.model.hidden_dim), fill_value=NO_OBJ_SCORE, dtype=torch.float32, device=inference_state["device"]),
            consolidated_mask_key: torch.full(size=(batch_size, 1, consolidated_H, consolidated_W), fill_value=NO_OBJ_SCORE, dtype=torch.float32, device=inference_state["storage_device"]),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                if run_mem_encoder:
                    if empty_mask_ptr is None: empty_mask_ptr = self.getemptymaskptr(inference_state, frame_idx)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                resized_obj_mask = F.interpolate(obj_mask, size=consolidated_pred_masks.shape[-2:], mode="bilinear", align_corners=False)
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = F.interpolate(consolidated_out["pred_masks"].to(device, non_blocking=True), size=(self.model.image_size, self.model.image_size), mode="bilinear", align_corners=False)
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model.applynonoverlappingconstraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self.runmemoryencoder(inference_state=inference_state, frame_idx=frame_idx, batch_size=batch_size, high_res_masks=high_res_masks, is_mask_from_pts=True)
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        return consolidated_out
    '''getemptymaskptr'''
    def getemptymaskptr(self, inference_state, frame_idx):
        batch_size = 1
        mask_inputs = torch.zeros((batch_size, 1, self.model.image_size, self.model.image_size), dtype=torch.float32, device=inference_state["device"])
        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.getimagefeature(inference_state, frame_idx, batch_size)
        current_out = self.model.trackstep(
            frame_idx=frame_idx, is_init_cond_frame=True, current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds, feat_sizes=feat_sizes,
            point_inputs=None, mask_inputs=mask_inputs, output_dict={}, num_frames=inference_state["num_frames"], track_in_reverse=False, run_mem_encoder=False, prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]
    '''propagateinvideopreflight'''
    @torch.inference_mode()
    def propagateinvideopreflight(self, inference_state):
        inference_state["tracking_has_started"] = True
        batch_size = self.getobjnum(inference_state)
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated_out = self.consolidatetempoutputacrossobj(inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True)
                output_dict[storage_key][frame_idx] = consolidated_out
                self.addoutputperobject(inference_state, frame_idx, consolidated_out, storage_key)
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1)
                if clear_non_cond_mem:
                    self.clearnoncondmemaroundinput(inference_state, frame_idx)
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
        all_consolidated_frame_inds = (consolidated_frame_inds["cond_frame_outputs"] | consolidated_frame_inds["non_cond_frame_outputs"])
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds
    '''propagateinvideo'''
    @torch.inference_mode()
    def propagateinvideo(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
        self.propagateinvideopreflight(inference_state)
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self.getobjnum(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first.")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1)
        if start_frame_idx is None:
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    self.clearnoncondmemaroundinput(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self.runsingleframeinference(
                    inference_state=inference_state, output_dict=output_dict, frame_idx=frame_idx, batch_size=batch_size, is_init_cond_frame=False, point_inputs=None, mask_inputs=None, reverse=reverse, run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out
            self.addoutputperobject(inference_state, frame_idx, current_out, storage_key)
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            _, video_res_masks = self.getorigvideoresoutput(inference_state, pred_masks)
            yield frame_idx, obj_ids, video_res_masks
    '''addoutputperobject'''
    def addoutputperobject(self, inference_state, frame_idx, current_out, storage_key):
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)
        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)
        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {"maskmem_features": None, "maskmem_pos_enc": None, "pred_masks": current_out["pred_masks"][obj_slice], "obj_ptr": current_out["obj_ptr"][obj_slice]}
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out
    '''resetstate'''
    @torch.inference_mode()
    def resetstate(self, inference_state):
        self.resettrackingresults(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
    '''resettrackingresults'''
    def resettrackingresults(self, inference_state):
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
    '''getimagefeature'''
    def getimagefeature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
            backbone_out = self.model.forwardimage(image)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {"backbone_fpn": backbone_out["backbone_fpn"].copy(), "vision_pos_enc": backbone_out["vision_pos_enc"].copy()}
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
        features = self.model.preparebackbonefeatures(expanded_backbone_out)
        features = (expanded_image,) + features
        return features
    '''runsingleframeinference'''
    def runsingleframeinference(self, inference_state, output_dict, frame_idx, batch_size, is_init_cond_frame, point_inputs, mask_inputs, reverse, run_mem_encoder, prev_sam_mask_logits=None):
        # retrieve correct image features
        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.getimagefeature(inference_state, frame_idx, batch_size)
        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.trackstep(
            frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame, current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes, point_inputs=point_inputs, mask_inputs=mask_inputs, output_dict=output_dict, num_frames=inference_state["num_frames"], track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder, prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fillholesinmaskscores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self.getmaskmemposenc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {"maskmem_features": maskmem_features, "maskmem_pos_enc": maskmem_pos_enc, "pred_masks": pred_masks, "obj_ptr": obj_ptr}
        # return
        return compact_current_out, pred_masks_gpu
    '''runmemoryencoder'''
    def runmemoryencoder(self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts):
        _, _, current_vision_feats, _, feat_sizes = self.getimagefeature(inference_state, frame_idx, batch_size)
        maskmem_features, maskmem_pos_enc = self.model.encodenewmemory(current_vision_feats=current_vision_feats, feat_sizes=feat_sizes, pred_masks_high_res=high_res_masks, is_mask_from_pts=is_mask_from_pts)
        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self.getmaskmemposenc(inference_state, {"maskmem_pos_enc": maskmem_pos_enc})
        # return
        return maskmem_features, maskmem_pos_enc
    '''getmaskmemposenc'''
    def getmaskmemposenc(self, inference_state, current_out):
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc
    '''clearnoncondmemaroundinput'''
    def clearnoncondmemaroundinput(self, inference_state, frame_idx):
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)