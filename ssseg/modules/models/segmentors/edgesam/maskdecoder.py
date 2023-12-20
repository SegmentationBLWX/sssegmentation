'''
Function:
    Implementation of MaskDecoder
Author:
    Zhenchao Jin
'''
from ..sam.maskdecoder import MaskDecoder as _MaskDecoder


'''MaskDecoder'''
class MaskDecoder(_MaskDecoder):
    '''forward'''
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, num_multimask_outputs):
        assert num_multimask_outputs in [1, 3, 4]
        masks, iou_pred = self.predictmasks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # select the correct mask or masks for output
        mask_slice = {
            1: slice(0, 1), 3: slice(1, None), 4: slice(0, None),
        }[num_multimask_outputs]
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # prepare output
        return masks, iou_pred