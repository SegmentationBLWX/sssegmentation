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
image1_pts = np.array([[[500, 375]], [[650, 750]]])
image1_labels = np.array([[1], [1]])
image2_pts = np.array([[[400, 300]], [[630, 300]]])
image2_labels = np.array([[1], [1]])
pts_batch = [image1_pts, image2_pts]
labels_batch = [image1_labels, image2_labels]
# set image
predictor.setimagebatch(img_batch)
# inference
masks_batch, scores_batch, _ = predictor.predictbatch(pts_batch, labels_batch, box_batch=None, multimask_output=True)
# select the best single mask per object
best_masks = []
for masks, scores in zip(masks_batch, scores_batch):
    best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])
# show results
for idx, (image, points, labels, masks) in enumerate(zip(img_batch, pts_batch, labels_batch, best_masks)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)   
    for mask in masks:
        showmask(mask, plt.gca(), random_color=True)
    showpoints(points, labels, plt.gca())
    plt.savefig(f'output_{idx}.png')