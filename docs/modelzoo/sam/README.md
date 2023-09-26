## Introduction

<a href="https://github.com/facebookresearch/segment-anything">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/sam/sam.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2304.02643.pdf">SAM (ArXiv'2023)</a></summary>

```latex
@article{kirillov2023segany,
    title={Segment Anything},
    author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
    journal={arXiv:2304.02643},
    year={2023}
}
```

</details>


## Inference with SAM

#### Environment Set-up

Install sssegmentation:

```sh
git clone https://github.com/SegmentationBLWX/sssegmentation
cd sssegmentation
pip install -r requirements.txt # install dependencies refer to https://sssegmentation.readthedocs.io/en/latest/GetStarted.html#prerequisites
```

Download images:

```sh
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg
```

Refer to [SAM official repo](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb), we provide some examples to use sssegmenation to perform SAM.

#### Selecting objects with SAM

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor

'''showmask'''
def showmask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 
'''showpoints''' 
def showpoints(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

'''showbox'''
def showbox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMPredictor(use_default_sam_h=True) or SAMPredictor(use_default_sam_l=True) or SAMPredictor(use_default_sam_b=True)
predictor = SAMPredictor(use_default_sam_h=True)
# set image
predictor.setimage(image)
# set prompt
input_label = np.array([1])
input_point = np.array([[500, 375]])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
# show results
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    showmask(mask, plt.gca())
    showpoints(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
```

#### Specifying a specific object with additional points

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor

'''showmask'''
def showmask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 
'''showpoints''' 
def showpoints(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

'''showbox'''
def showbox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMPredictor(use_default_sam_h=True) or SAMPredictor(use_default_sam_l=True) or SAMPredictor(use_default_sam_b=True)
predictor = SAMPredictor(use_default_sam_h=True)
# set image
predictor.setimage(image)
# set prompt
input_label = np.array([1])
input_point = np.array([[500, 375]])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
# set prompt for the second time
input_label = np.array([1, 1])
input_point = np.array([[500, 375], [1125, 625]])
# inference for the second time
mask_input = logits[np.argmax(scores), :, :]
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
# show results
plt.figure(figsize=(10,10))
plt.imshow(image)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
```

To exclude the car and specify just the window, a background point (with label 0, here shown in red) can be supplied.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor

'''showmask'''
def showmask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 
'''showpoints''' 
def showpoints(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

'''showbox'''
def showbox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMPredictor(use_default_sam_h=True) or SAMPredictor(use_default_sam_l=True) or SAMPredictor(use_default_sam_b=True)
predictor = SAMPredictor(use_default_sam_h=True)
# set image
predictor.setimage(image)
# set prompt
input_label = np.array([1])
input_point = np.array([[500, 375]])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
# set prompt for the second time
input_label = np.array([1, 0])
input_point = np.array([[500, 375], [1125, 625]])
# inference for the second time
mask_input = logits[np.argmax(scores), :, :]
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
# show results
plt.figure(figsize=(10,10))
plt.imshow(image)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
```

#### Specifying a specific object with a box


#### Combining points and boxes


#### Batched prompt inputs


#### End-to-end batched inference