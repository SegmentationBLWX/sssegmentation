'''
Function:
    Implementation of SAMHQ
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
from .maskdecoder import MaskDecoderHQ
from torchvision.ops.boxes import batched_nms, box_area
from ..sam import SAM, SAMPredictor, SAMAutomaticMaskGenerator
from ..sam.amg import (
    MaskData, isboxnearcropedge, boxxyxytoxywh, batchiterator, masktorlepytorch, rletomask, areafromrle, calculatestabilityscore, 
    generatecropboxes, uncropboxesxyxy, uncroppoints, uncropmasks, cocoencoderle, batchedmasktobox
)


'''SAMHQ'''
class SAMHQ(SAM):
    mask_threshold = 0.0
    image_format = 'RGB'
    def __init__(self, cfg, mode):
        vit_dim = cfg['head'].pop('vit_dim')
        super(SAMHQ, self).__init__(cfg=cfg, mode=mode)
        cfg['head']['vit_dim'] = vit_dim
        self.mask_decoder = MaskDecoderHQ(**cfg['head'])
    '''inference'''
    @torch.no_grad()
    def inference(self, batched_input, multimask_output=False, hq_token_only=False):
        input_images = torch.stack([self.preprocess(x['image']) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images, return_interm_embeddings=True)
        interm_embeddings = interm_embeddings[0]
        outputs = []
        for image_record, curr_embedding, curr_interm in zip(batched_input, image_embeddings, interm_embeddings):
            if 'point_coords' in image_record:
                points = (image_record['point_coords'], image_record['point_labels'])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points, boxes=image_record.get('boxes', None), masks=image_record.get('mask_inputs', None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0), image_pe=self.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, 
                multimask_output=multimask_output, hq_token_only=hq_token_only, interm_embeddings=curr_interm.unsqueeze(0).unsqueeze(0),
            )
            masks = self.postprocessmasks(
                low_res_masks, input_size=image_record['image'].shape[-2:], original_size=image_record['original_size'],
            )
            masks = masks > self.mask_threshold
            outputs.append({
                'masks': masks, 'iou_predictions': iou_predictions, 'low_res_logits': low_res_masks,
            })
        return outputs


'''SAMHQPredictor'''
class SAMHQPredictor(SAMPredictor):
    def __init__(self, sam_cfg=None, use_default_samhq_t_5m=False, use_default_samhq_b=False, use_default_samhq_l=False, use_default_samhq_h=False, device='cuda', load_ckpt_strict=True):
        if sam_cfg is None:
            sam_cfg = {
                'backbone': {
                    'depth': None, 'embed_dim': None, 'img_size': 1024, 'mlp_ratio': 4, 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-6}, 'num_heads': None, 
                    'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'global_attn_indexes': None, 'window_size': 14, 'out_chans': 256, 'type': 'SAMViT'
                },
                'prompt': {
                    'embed_dim': 256, 'image_embedding_size': (1024//16, 1024//16), 'input_image_size': (1024, 1024), 'mask_in_chans': 16,
                },
                'head': {
                    'num_multimask_outputs': 3, 'transformer_cfg': {'depth': 2, 'embedding_dim': 256, 'mlp_dim': 2048, 'num_heads': 8}, 
                    'transformer_dim': 256, 'iou_head_depth': 3, 'iou_head_hidden_dim': 256, 'vit_dim': None,
                },
            }
            if use_default_samhq_h:
                assert (not use_default_samhq_b) and (not use_default_samhq_l) and (not use_default_samhq_t_5m)
                sam_cfg['backbone']['depth'] = 32
                sam_cfg['backbone']['embed_dim'] = 1280
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [7, 15, 23, 31]
                sam_cfg['head']['vit_dim'] = 1280
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth'
            if use_default_samhq_l:
                assert (not use_default_samhq_b) and (not use_default_samhq_h) and (not use_default_samhq_t_5m)
                sam_cfg['backbone']['depth'] = 24
                sam_cfg['backbone']['embed_dim'] = 1024
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [5, 11, 17, 23]
                sam_cfg['head']['vit_dim'] = 1024
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth'
            if use_default_samhq_b:
                assert (not use_default_samhq_l) and (not use_default_samhq_h) and (not use_default_samhq_t_5m)
                sam_cfg['backbone']['depth'] = 12
                sam_cfg['backbone']['embed_dim'] = 768
                sam_cfg['backbone']['num_heads'] = 12
                sam_cfg['backbone']['global_attn_indexes'] = [2, 5, 8, 11]
                sam_cfg['head']['vit_dim'] = 768
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth'
            if use_default_samhq_t_5m:
                assert (not use_default_samhq_b) and (not use_default_samhq_l) and (not use_default_samhq_h)
                sam_cfg['backbone'] = {
                    'structure_type': 'tiny_vit_5m_22kto1k_distill', 'img_size': 1024, 'in_chans': 3, 'embed_dims': [64, 128, 160, 320], 'depths': [2, 2, 6, 2], 
                    'num_heads': [2, 4, 5, 10], 'window_sizes': [7, 7, 14, 7], 'mlp_ratio': 4., 'drop_rate': 0., 'drop_path_rate': 0.0, 'use_checkpoint': False, 
                    'mbconv_expand_ratio': 4.0, 'local_conv_size': 3, 'pretrained': False, 'pretrained_model_path': '', 'type': 'MobileSAMTinyViT'
                }
                sam_cfg['head']['vit_dim'] = 160
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth'
                load_ckpt_strict = False
        else:
            assert (not use_default_samhq_b) and (not use_default_samhq_l) and (not use_default_samhq_h) and (not use_default_samhq_t_5m)
        super(SAMHQPredictor, self).__init__(
            use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, sam_cfg=sam_cfg, device=device, load_ckpt_strict=load_ckpt_strict,
        )
        self.model.eval()
    '''buildsam'''
    def buildsam(self, sam_cfg, device):
        sam_model = SAMHQ(sam_cfg, mode='TEST')
        sam_model.to(device=device)
        sam_model.eval()
        return sam_model
    '''settorchimage'''
    @torch.no_grad()
    def settorchimage(self, transformed_image, original_image_size):
        assert (
            len(transformed_image.shape) == 4 and transformed_image.shape[1] == 3 and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.resetimage()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features, self.interm_features = self.model.image_encoder(input_image, return_interm_embeddings=True)
        self.is_image_set = True
    '''predict'''
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False, hq_token_only=False):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .setimage(...) before mask prediction')
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
            coords_torch, labels_torch, box_torch, mask_input_torch, multimask_output, return_logits=return_logits, hq_token_only=hq_token_only,
        )
        # return result
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    '''predicttorch'''
    @torch.no_grad()
    def predicttorch(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output=True, return_logits=False, hq_token_only=False):
        if not self.is_image_set:
            raise RuntimeError("an image must be set with .setimage(...) before mask prediction.")
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
            dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, hq_token_only=hq_token_only, interm_embeddings=self.interm_features,
        )
        # upscale the masks to the original image resolution
        masks = self.model.postprocessmasks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        # return
        return masks, iou_predictions, low_res_masks


'''SAMHQAutomaticMaskGenerator'''
class SAMHQAutomaticMaskGenerator(SAMAutomaticMaskGenerator):
    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, device='cuda',
                 box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1, point_grids=None,
                 min_mask_region_area=0, output_mode='binary_mask', sam_cfg=None, use_default_samhq_t_5m=False, use_default_samhq_b=False, use_default_samhq_l=False, 
                 use_default_samhq_h=False, load_ckpt_strict=True):
        user_defined_sam_predictor = SAMHQPredictor(
            sam_cfg=sam_cfg, use_default_samhq_t_5m=use_default_samhq_t_5m, use_default_samhq_b=use_default_samhq_b, use_default_samhq_l=use_default_samhq_l,
            use_default_samhq_h=use_default_samhq_h, device=device, load_ckpt_strict=load_ckpt_strict
        )
        super(SAMHQAutomaticMaskGenerator, self).__init__(
            points_per_side=points_per_side, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, 
            stability_score_offset=stability_score_offset, device=device, box_nms_thresh=box_nms_thresh, crop_n_layers=crop_n_layers, crop_nms_thresh=crop_nms_thresh, 
            crop_overlap_ratio=crop_overlap_ratio, crop_n_points_downscale_factor=crop_n_points_downscale_factor, point_grids=point_grids, min_mask_region_area=min_mask_region_area, 
            output_mode=output_mode, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, user_defined_sam_predictor=user_defined_sam_predictor,
            load_ckpt_strict=load_ckpt_strict,
        )
    '''generate'''
    @torch.no_grad()
    def generate(self, image, hq_token_only=False):
        # generate masks
        mask_data = self.generatemasks(image, hq_token_only)
        # filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocesssmallregions(mask_data, self.min_mask_region_area, max(self.box_nms_thresh, self.crop_nms_thresh))
        # encode masks
        if self.output_mode == 'coco_rle':
            mask_data['segmentations'] = [cocoencoderle(rle) for rle in mask_data['rles']]
        elif self.output_mode == 'binary_mask':
            mask_data['segmentations'] = [rletomask(rle) for rle in mask_data['rles']]
        else:
            mask_data['segmentations'] = mask_data['rles']
        # write mask records
        curr_anns = []
        for idx in range(len(mask_data['segmentations'])):
            ann = {
                'segmentation': mask_data['segmentations'][idx], 'area': areafromrle(mask_data['rles'][idx]),
                'bbox': boxxyxytoxywh(mask_data['boxes'][idx]).tolist(),
                'predicted_iou': mask_data['iou_preds'][idx].item(),
                'point_coords': [mask_data['points'][idx].tolist()],
                'stability_score': mask_data['stability_score'][idx].item(),
                'crop_box': boxxyxytoxywh(mask_data['crop_boxes'][idx]).tolist(),
            }
            curr_anns.append(ann)
        # return
        return curr_anns
    '''generatemasks'''
    def generatemasks(self, image, hq_token_only=False):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generatecropboxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
        # iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self.processcrop(image, crop_box, layer_idx, orig_size, hq_token_only)
            data.cat(crop_data)
        # remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # prefer masks from smaller crops
            scores = 1 / box_area(data['crop_boxes'])
            scores = scores.to(data['boxes'].device)
            keep_by_nms = batched_nms(data['boxes'].float(), scores, torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.crop_nms_thresh)
            data.filter(keep_by_nms)
        # return
        data.tonumpy()
        return data
    '''processcrop'''
    def processcrop(self, image, crop_box, crop_layer_idx, orig_size, hq_token_only):
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
            batch_data = self.processbatch(points, cropped_im_size, crop_box, orig_size, hq_token_only)
            data.cat(batch_data)
            del batch_data
        self.predictor.resetimage()
        # remove duplicates within this crop.
        keep_by_nms = batched_nms(data['boxes'].float(), data['iou_preds'], torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.box_nms_thresh)
        data.filter(keep_by_nms)
        # return to the original image frame
        data['boxes'] = uncropboxesxyxy(data['boxes'], crop_box)
        data['points'] = uncroppoints(data['points'], crop_box)
        data['crop_boxes'] = torch.tensor([crop_box for _ in range(len(data['rles']))])
        # return
        return data
    '''processbatch'''
    def processbatch(self, points, im_size, crop_box, orig_size, hq_token_only):
        orig_h, orig_w = orig_size
        # run model on this batch
        transformed_points = self.predictor.transform.applycoords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predicttorch(in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True, hq_token_only=hq_token_only)
        # serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks
        # filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data['iou_preds'] > self.pred_iou_thresh
            data.filter(keep_mask)
        # calculate stability score
        data['stability_score'] = calculatestabilityscore(
            data['masks'], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data['stability_score'] >= self.stability_score_thresh
            data.filter(keep_mask)
        # threshold masks and calculate boxes
        data['masks'] = data['masks'] > self.predictor.model.mask_threshold
        data['boxes'] = batchedmasktobox(data['masks'])
        # filter boxes that touch crop boundaries
        keep_mask = ~isboxnearcropedge(data['boxes'], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        # compress to RLE
        data['masks'] = uncropmasks(data['masks'], crop_box, orig_h, orig_w)
        data['rles'] = masktorlepytorch(data['masks'])
        del data['masks']
        # return
        return data