'''
Function:
    SAMV2 examples: End-to-end batched inference
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
image1 = Image.open('images/truck.jpg')
image1 = np.array(image1.convert("RGB"))
image2 = Image.open('images/groceries.jpg')
image2 = np.array(image2.convert("RGB"))
img_batch = [image1, image2]


# predictor could be SAMV2ImagePredictor(use_default_samv2_t=True) or SAMV2ImagePredictor(use_default_samv2_s=True) or SAMV2ImagePredictor(use_default_samv2_bplus=True) or SAMV2ImagePredictor(use_default_samv2_l=True)
predictor = SAMV2ImagePredictor(use_default_samv2_l=True, device='cuda')
# set prompt
image1_boxes = np.array([[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]])
image2_boxes = np.array([[450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350]])
boxes_batch = [image1_boxes, image2_boxes]
# set image
predictor.setimagebatch(img_batch)
# inference
masks_batch, scores_batch, _ = predictor.predictbatch(None, None, box_batch=boxes_batch, multimask_output=False)
# show results
for idx, (image, boxes, masks) in enumerate(zip(img_batch, boxes_batch, masks_batch)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)   
    for mask in masks:
        showmask(mask.squeeze(0), plt.gca(), random_color=True)
    for box in boxes:
        showbox(box, plt.gca())
    plt.savefig(f'output_{idx}.png')