'''
Function:
    SAMV2 examples: Selecting objects with SAMV2
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
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
input_point = np.array([[500, 375]])
input_label = np.array([1])
# inference
masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]
# show results
showmasks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)