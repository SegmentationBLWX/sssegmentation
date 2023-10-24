'''
Function:
    Implementation of MobileSAM
Author:
    Zhenchao Jin
'''
from ..sam import SAM, SAMPredictor, SAMAutomaticMaskGenerator


'''MobileSAM'''
class MobileSAM(SAM):
    mask_threshold = 0.0
    image_format = 'RGB'
    def __init__(self, cfg, mode):
        super(MobileSAM, self).__init__(cfg=cfg, mode=mode)


'''MobileSAMPredictor'''
class MobileSAMPredictor(SAMPredictor):
    def __init__(self, sam_cfg=None, use_default_sam_t_5m=False, device='cuda', load_ckpt_strict=False):
        if sam_cfg is None:
            sam_cfg = {
                'backbone': {
                    'structure_type': 'tiny_vit_5m_22kto1k_distill', 'img_size': 1024, 'in_chans': 3, 'embed_dims': [64, 128, 160, 320], 'depths': [2, 2, 6, 2], 
                    'num_heads': [2, 4, 5, 10], 'window_sizes': [7, 7, 14, 7], 'mlp_ratio': 4., 'drop_rate': 0., 'drop_path_rate': 0.0, 'use_checkpoint': False, 
                    'mbconv_expand_ratio': 4.0, 'local_conv_size': 3, 'pretrained': False, 'pretrained_model_path': '', 'type': 'MobileSAMTinyViT'
                },
                'prompt': {
                    'embed_dim': 256, 'image_embedding_size': (1024//16, 1024//16), 'input_image_size': (1024, 1024), 'mask_in_chans': 16,
                },
                'head': {
                    'num_multimask_outputs': 3, 'transformer_cfg': {'depth': 2, 'embedding_dim': 256, 'mlp_dim': 2048, 'num_heads': 8}, 
                    'transformer_dim': 256, 'iou_head_depth': 3, 'iou_head_hidden_dim': 256,
                },
            }
            if use_default_sam_t_5m:
                sam_cfg['ckptpath'] = 'https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilesam/sam_vit_t_5m.pth'
        else:
            assert (not use_default_sam_t_5m)
        super(MobileSAMPredictor, self).__init__(
            use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, sam_cfg=sam_cfg, device=device, load_ckpt_strict=load_ckpt_strict,
        )
        self.model.eval()
    '''buildsam'''
    def buildsam(self, sam_cfg, device):
        sam_model = MobileSAM(sam_cfg, mode='TEST')
        sam_model.to(device=device)
        sam_model.eval()
        return sam_model


'''MobileSAMAutomaticMaskGenerator'''
class MobileSAMAutomaticMaskGenerator(SAMAutomaticMaskGenerator):
    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, device='cuda',
                 box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1, point_grids=None,
                 min_mask_region_area=0, output_mode='binary_mask', sam_cfg=None, use_default_sam_t_5m=False, load_ckpt_strict=False):
        user_defined_sam_predictor = MobileSAMPredictor(sam_cfg=sam_cfg, use_default_sam_t_5m=use_default_sam_t_5m, device=device, load_ckpt_strict=load_ckpt_strict)
        super(MobileSAMAutomaticMaskGenerator, self).__init__(
            points_per_side=points_per_side, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, 
            stability_score_offset=stability_score_offset, device=device, box_nms_thresh=box_nms_thresh, crop_n_layers=crop_n_layers, crop_nms_thresh=crop_nms_thresh, 
            crop_overlap_ratio=crop_overlap_ratio, crop_n_points_downscale_factor=crop_n_points_downscale_factor, point_grids=point_grids, min_mask_region_area=min_mask_region_area, 
            output_mode=output_mode, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, user_defined_sam_predictor=user_defined_sam_predictor,
            load_ckpt_strict=load_ckpt_strict,
        )