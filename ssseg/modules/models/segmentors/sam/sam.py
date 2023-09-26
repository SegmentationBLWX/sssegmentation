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
import torch.utils.model_zoo as model_zoo
from functools import partial
from .maskdecoder import MaskDecoder
from ...backbones import BuildBackbone
from .promptencoder import PromptEncoder
from .transforms import ResizeLongestSide


'''SAM'''
class SAM(nn.Module):
    mask_threshold = 0.0
    image_format = "RGB"
    def __init__(self, cfg, mode):
        super(SAM, self).__init__()
        assert mode in ['TEST'], 'only support test mode for SAM now'
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
    def forward(self, x, targets=None):
        raise NotImplementedError('train SAM not to be implemented')
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
                image_embeddings=curr_embedding.unsqueeze(0), image_pe=self.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings,
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
    def __init__(self, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False):
        super(SAMPredictor, self).__init__()
        if sam_cfg is None:
            default_sam_cfg = {
                'backbone': {
                    'depth': None, 'embed_dim': None, 'img_size': 1024, 'mlp_ratio': 4, 'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6), 'num_heads': None, 
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
        self.model = self.buildsam(sam_cfg)
        if 'ckptpath' in sam_cfg and (os.path.exists(sam_cfg['ckptpath']) or sam_cfg['ckptpath'].startswith('https')):
            if os.path.exists(sam_cfg['ckptpath']):
                with open(sam_cfg['ckptpath'], 'rb') as fp:
                    state_dict = torch.load(fp)
            elif sam_cfg['ckptpath'].startswith('https'):
                state_dict = model_zoo.load_url(sam_cfg['ckptpath'])
            else:
                raise ValueError('ckptpath %s could not be loaded' % sam_cfg['ckptpath'])
            self.model.load_state_dict(state_dict, strict=True)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.resetimage()
    '''buildsam'''
    def buildsam(self, sam_cfg):
        sam_model = SAM(sam_cfg, mode='TEST')
        sam_model.eval()
        return sam_model
    '''setimage'''
    def setimage(self, image, image_format='RGB'):
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]
        # transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
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
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        masks, iou_predictions, low_res_masks = self.predicttorch(
            coords_torch, labels_torch, box_torch, mask_input_torch, multimask_output, return_logits=return_logits,
        )
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
            image_embeddings=self.features, image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output,
        )
        # upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
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