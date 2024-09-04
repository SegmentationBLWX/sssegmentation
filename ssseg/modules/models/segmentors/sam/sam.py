'''
Function:
    Implementation of SAM
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from ..base import BaseSegmentor
from .maskdecoder import MaskDecoder
from ...backbones import BuildBackbone
from .promptencoder import PromptEncoder
from .transforms import ResizeLongestSide
from torchvision.ops.boxes import batched_nms, box_area
from .amg import (
    MaskData, isboxnearcropedge, boxxyxytoxywh, batchiterator, masktorlepytorch, rletomask, areafromrle, calculatestabilityscore, buildpointgrid, 
    buildalllayerpointgrids, generatecropboxes, uncropboxesxyxy, uncroppoints, uncropmasks, removesmallregions, cocoencoderle, batchedmasktobox
)


'''SAM'''
class SAM(BaseSegmentor):
    mask_threshold = 0.0
    image_format = 'RGB'
    def __init__(self, cfg, mode):
        backbone = cfg.pop('backbone')
        super(SAM, self).__init__(cfg=cfg, mode=mode)
        cfg['backbone'] = backbone
        assert mode in ['TEST'], f'only support TEST mode for {self.__class__.__name__}'
        pixel_mean = cfg.get('pixel_mean', [123.675, 116.28, 103.53])
        pixel_std = cfg.get('pixel_std', [58.395, 57.12, 57.375])
        self.image_encoder = BuildBackbone(cfg['backbone'])
        self.prompt_encoder = PromptEncoder(**cfg['prompt'])
        self.mask_decoder = MaskDecoder(**cfg['head'])
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)
    '''device'''
    @property
    def device(self):
        return self.pixel_mean.device
    '''forward'''
    def forward(self, data_meta):
        raise NotImplementedError(f'train {self.__class__.__name__} not to be implemented')
    '''inference'''
    @torch.no_grad()
    def inference(self, batched_input, multimask_output=False):
        input_images = torch.stack([self.preprocess(x['image']) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if 'point_coords' in image_record:
                points = (image_record['point_coords'], image_record['point_labels'])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points, boxes=image_record.get('boxes', None), masks=image_record.get('mask_inputs', None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0), image_pe=self.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output,
            )
            masks = self.postprocessmasks(
                low_res_masks, input_size=image_record['image'].shape[-2:], original_size=image_record['original_size'],
            )
            masks = masks > self.mask_threshold
            outputs.append({
                'masks': masks, 'iou_predictions': iou_predictions, 'low_res_logits': low_res_masks,
            })
        return outputs
    '''postprocessmasks'''
    def postprocessmasks(self, masks, input_size, original_size):
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode='bilinear', align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode='bilinear', align_corners=False)
        return masks
    '''preprocess'''
    def preprocess(self, x):
        # normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


'''SAMPredictor'''
class SAMPredictor(nn.Module):
    def __init__(self, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, device='cuda', load_ckpt_strict=True):
        super(SAMPredictor, self).__init__()
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
                    'transformer_dim': 256, 'iou_head_depth': 3, 'iou_head_hidden_dim': 256,
                },
            }
            if use_default_sam_h:
                assert (not use_default_sam_l) and (not use_default_sam_b)
                sam_cfg['backbone']['depth'] = 32
                sam_cfg['backbone']['embed_dim'] = 1280
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [7, 15, 23, 31]
                sam_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            if use_default_sam_l:
                assert (not use_default_sam_h) and (not use_default_sam_b)
                sam_cfg['backbone']['depth'] = 24
                sam_cfg['backbone']['embed_dim'] = 1024
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [5, 11, 17, 23]
                sam_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
            if use_default_sam_b:
                assert (not use_default_sam_h) and (not use_default_sam_l)
                sam_cfg['backbone']['depth'] = 12
                sam_cfg['backbone']['embed_dim'] = 768
                sam_cfg['backbone']['num_heads'] = 12
                sam_cfg['backbone']['global_attn_indexes'] = [2, 5, 8, 11]
                sam_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        else:
            assert (not use_default_sam_h) and (not use_default_sam_l) and (not use_default_sam_b)
        self.model = self.buildsam(sam_cfg, device)
        if 'ckptpath' in sam_cfg and (os.path.exists(sam_cfg['ckptpath']) or sam_cfg['ckptpath'].startswith('https')):
            if os.path.exists(sam_cfg['ckptpath']):
                with open(sam_cfg['ckptpath'], 'rb') as fp:
                    state_dict = torch.load(fp)
            elif sam_cfg['ckptpath'].startswith('https'):
                state_dict = model_zoo.load_url(sam_cfg['ckptpath'])
            else:
                raise ValueError('ckptpath %s could not be loaded' % sam_cfg['ckptpath'])
            self.model.load_state_dict(state_dict, strict=load_ckpt_strict)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.resetimage()
    '''buildsam'''
    def buildsam(self, sam_cfg, device):
        sam_model = SAM(sam_cfg, mode='TEST')
        sam_model.to(device=device)
        sam_model.eval()
        return sam_model
    '''setimage'''
    def setimage(self, image, image_format='RGB'):
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]
        # transform the image to the form expected by the model
        input_image = self.transform.applyimage(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.settorchimage(input_image_torch, image.shape[:2])
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
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True
    '''predict'''
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
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
            coords_torch, labels_torch, box_torch, mask_input_torch, multimask_output, return_logits=return_logits,
        )
        # return result
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    '''predicttorch'''
    @torch.no_grad()
    def predicttorch(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output=True, return_logits=False):
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
            dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output,
        )
        # upscale the masks to the original image resolution
        masks = self.model.postprocessmasks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        # return
        return masks, iou_predictions, low_res_masks
    '''getimageembedding'''
    def getimageembedding(self):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .set_image(...) to generate an embedding.')
        assert self.features is not None, 'features must exist if an image has been set.'
        return self.features
    '''device'''
    @property
    def device(self):
        return self.model.device
    '''resetimage'''
    def resetimage(self):
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


'''SAMAutomaticMaskGenerator'''
class SAMAutomaticMaskGenerator(nn.Module):
    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, device='cuda',
                 box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1, point_grids=None,
                 min_mask_region_area=0, output_mode='binary_mask', sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False,
                 user_defined_sam_predictor=None, load_ckpt_strict=True):
        super(SAMAutomaticMaskGenerator, self).__init__()
        # assert arguments
        assert (points_per_side is None) != (point_grids is None), 'exactly one of points_per_side or point_grid must be provided.'
        assert output_mode in ['binary_mask', 'uncompressed_rle', 'coco_rle'], f'unknown output_mode {output_mode}.'
        # set point_grids
        if points_per_side is not None:
            self.point_grids = buildalllayerpointgrids(points_per_side, crop_n_layers, crop_n_points_downscale_factor)
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("can't have both points_per_side and point_grid be None")
        # set attributes
        if user_defined_sam_predictor is not None:
            self.predictor = user_defined_sam_predictor
        else:
            self.predictor = SAMPredictor(sam_cfg, use_default_sam_h, use_default_sam_l, use_default_sam_b, device=device, load_ckpt_strict=load_ckpt_strict)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
    '''generate'''
    @torch.no_grad()
    def generate(self, image):
        # generate masks
        mask_data = self.generatemasks(image)
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
            # prefer masks from smaller crops
            scores = 1 / box_area(data['crop_boxes'])
            scores = scores.to(data['boxes'].device)
            keep_by_nms = batched_nms(data['boxes'].float(), scores, torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.crop_nms_thresh)
            data.filter(keep_by_nms)
        # return
        data.tonumpy()
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
            batch_data = self.processbatch(points, cropped_im_size, crop_box, orig_size)
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
    def processbatch(self, points, im_size, crop_box, orig_size):
        orig_h, orig_w = orig_size
        # run model on this batch
        transformed_points = self.predictor.transform.applycoords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predicttorch(in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True)
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
    '''postprocesssmallregions'''
    @staticmethod
    def postprocesssmallregions(mask_data, min_area, nms_thresh):
        if len(mask_data['rles']) == 0: return mask_data
        # filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rletomask(rle)
            mask, changed = removesmallregions(mask, min_area, mode='holes')
            unchanged = not changed
            mask, changed = removesmallregions(mask, min_area, mode='islands')
            unchanged = unchanged and not changed
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # give score=0 to changed masks and score=1 to unchanged masks so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))
        # recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batchedmasktobox(masks)
        keep_by_nms = batched_nms(boxes.float(), torch.as_tensor(scores), torch.zeros_like(boxes[:, 0]), iou_threshold=nms_thresh)
        # only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data['rles'][i_mask] = masktorlepytorch(mask_torch)[0]
                mask_data['boxes'][i_mask] = boxes[i_mask]
        mask_data.filter(keep_by_nms)
        # return
        return mask_data