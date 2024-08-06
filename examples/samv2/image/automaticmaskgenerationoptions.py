'''
Function:
    SAMV2 examples: Automatic mask generation
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2.visualization import showanns
from ssseg.modules.models.segmentors.samv2 import SAMV2AutomaticMaskGenerator


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# read image
image = Image.open('images/cars.jpg')
image = np.array(image.convert("RGB"))


# mask_generator could be SAMV2AutomaticMaskGenerator(use_default_samv2_t=True) or SAMV2AutomaticMaskGenerator(use_default_samv2_s=True) or SAMV2AutomaticMaskGenerator(use_default_samv2_bplus=True) or SAMV2AutomaticMaskGenerator(use_default_samv2_l=True)
mask_generator = SAMV2AutomaticMaskGenerator(
    use_default_samv2_l=True, device='cuda', apply_postprocessing=False, points_per_side=64, points_per_batch=128, pred_iou_thresh=0.7, stability_score_thresh=0.92,
    stability_score_offset=0.7, crop_n_layers=1, box_nms_thresh=0.7, crop_n_points_downscale_factor=2, min_mask_region_area=25.0, use_m2m=True,
)
# generate
masks = mask_generator.generate(image)
# show results
print(len(masks))
print(masks[0].keys())
plt.figure(figsize=(20, 20))
plt.imshow(image)
showanns(masks)
plt.axis('off')
plt.savefig('output.png') 