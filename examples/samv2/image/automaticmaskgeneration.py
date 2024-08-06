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
mask_generator = SAMV2AutomaticMaskGenerator(use_default_samv2_l=True, device='cuda', apply_postprocessing=False)
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