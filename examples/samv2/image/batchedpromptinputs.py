'''
Function:
    SAMV2 examples: Batched prompt inputs
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2ImagePredictor
from ssseg.modules.models.segmentors.samv2.visualization import showmask, showpoints, showbox, showmasks


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# read image
image = Image.open('images/truck.jpg')
image = np.array(image.convert("RGB"))


# predictor could be SAMV2ImagePredictor(use_default_samv2_t=True) or SAMV2ImagePredictor(use_default_samv2_s=True) or SAMV2ImagePredictor(use_default_samv2_bplus=True) or SAMV2ImagePredictor(use_default_samv2_l=True)
predictor = SAMV2ImagePredictor(use_default_samv2_l=True, device='cuda')
# set image
predictor.setimage(image)
# set prompt
input_boxes = np.array([[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]])
# inference
masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=input_boxes, multimask_output=False)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    showmask(mask.squeeze(0), plt.gca(), random_color=True)
for box in input_boxes:
    showbox(box, plt.gca())
plt.axis('off')
plt.savefig('output.png')