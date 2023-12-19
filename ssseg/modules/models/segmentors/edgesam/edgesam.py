'''
Function:
    Implementation of EdgeSAM
Author:
    Zhenchao Jin
'''
import torch
from .maskdecoder import MaskDecoder
from ..sam.amg import calculatestabilityscore
from ..sam import SAM, SAMPredictor, SAMAutomaticMaskGenerator


'''EdgeSAM'''
class EdgeSAM(SAM):
    mask_threshold = 0.0
    image_format = 'RGB'
    def __init__(self, cfg, mode):
        super(EdgeSAM, self).__init__(cfg=cfg, mode=mode)
        self.mask_decoder = MaskDecoder(**cfg['head'])


'''EdgeSAMPredictor'''
class EdgeSAMPredictor(SAMPredictor):
    def __init__(self, sam_cfg=None, use_default_edgesam=False, use_default_edgesam_3x=False, device='cuda', load_ckpt_strict=True, stability_score_offset=1.0):
        if sam_cfg is None:
            sam_cfg = {
                'backbone': {
                    'type': 'EdgeSAMRepViT', 'structure_type': 'repvit_m1', 'arch': 'm1', 'img_size': 1024, 'upsample_mode': 'bicubic',
                },
                'prompt': {
                    'embed_dim': 256, 'image_embedding_size': (1024//16, 1024//16), 'input_image_size': (1024, 1024), 'mask_in_chans': 16,
                },
                'head': {
                    'num_multimask_outputs': 3, 'transformer_cfg': {'depth': 2, 'embedding_dim': 256, 'mlp_dim': 2048, 'num_heads': 8}, 
                    'transformer_dim': 256, 'iou_head_depth': 3, 'iou_head_hidden_dim': 256,
                },
            }
            if use_default_edgesam:
                assert (not use_default_edgesam_3x)
                sam_cfg['ckptpath'] = 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_edgesam/edge_sam.pth'
            if use_default_edgesam_3x:
                assert (not use_default_edgesam)
                sam_cfg['ckptpath'] = 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_edgesam/edge_sam_3x.pth'
        else:
            assert (not use_default_edgesam) and (not use_default_edgesam_3x)
        super(EdgeSAMPredictor, self).__init__(
            use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, sam_cfg=sam_cfg, device=device, load_ckpt_strict=load_ckpt_strict,
        )
        self.model.eval()
        self.stability_score_offset = stability_score_offset
    '''buildsam'''
    def buildsam(self, sam_cfg, device):
        sam_model = EdgeSAM(sam_cfg, mode='TEST')
        sam_model.to(device=device)
        sam_model.eval()
        return sam_model
    '''predict'''
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, num_multimask_outputs=3, return_logits=False, use_stability_score=False):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .set_image(...) before mask prediction')
        # transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, 'point_labels must be supplied if point_coords is supplied.'
            point_coords = self.transform.applycoords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.applyboxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        # predict
        masks, iou_predictions, low_res_masks = self.predicttorch(
            coords_torch, labels_torch, box_torch, mask_input_torch, num_multimask_outputs, return_logits=return_logits, use_stability_score=use_stability_score,
        )
        # return result
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    '''predicttorch'''
    @torch.no_grad()
    def predicttorch(self, point_coords, point_labels, boxes=None, mask_input=None, num_multimask_outputs=3, return_logits=False, use_stability_score=True):
        if not self.is_image_set:
            raise RuntimeError("an image must be set with .set_image(...) before mask prediction.")
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        # embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points, boxes=boxes, masks=mask_input,
        )
        # predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features, image_pe=self.model.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, num_multimask_outputs=num_multimask_outputs,
        )
        if use_stability_score:
            iou_predictions = calculatestabilityscore(
                low_res_masks, self.model.mask_threshold, self.stability_score_offset
            )
        # upscale the masks to the original image resolution
        masks = self.model.postprocessmasks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        # return
        return masks, iou_predictions, low_res_masks


'''EdgeSAMAutomaticMaskGenerator'''
class EdgeSAMAutomaticMaskGenerator(SAMAutomaticMaskGenerator):
    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, device='cuda',
                 box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1, point_grids=None,
                 min_mask_region_area=0, output_mode='binary_mask', sam_cfg=None, use_default_sam_t_5m=False, load_ckpt_strict=False):
        user_defined_sam_predictor = EdgeSAMPredictor(sam_cfg=sam_cfg, use_default_sam_t_5m=use_default_sam_t_5m, device=device, load_ckpt_strict=load_ckpt_strict)
        super(EdgeSAMAutomaticMaskGenerator, self).__init__(
            points_per_side=points_per_side, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, 
            stability_score_offset=stability_score_offset, device=device, box_nms_thresh=box_nms_thresh, crop_n_layers=crop_n_layers, crop_nms_thresh=crop_nms_thresh, 
            crop_overlap_ratio=crop_overlap_ratio, crop_n_points_downscale_factor=crop_n_points_downscale_factor, point_grids=point_grids, min_mask_region_area=min_mask_region_area, 
            output_mode=output_mode, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, user_defined_sam_predictor=user_defined_sam_predictor,
            load_ckpt_strict=load_ckpt_strict,
        )