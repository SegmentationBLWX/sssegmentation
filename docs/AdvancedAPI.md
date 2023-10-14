# Advanced API


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
input_box = np.array([425, 600, 700, 875])
# inference
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.show()
```

#### Combining points and boxes

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
input_label = np.array([0])
input_point = np.array([[575, 750]])
input_box = np.array([425, 600, 700, 875])
# inference
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
```

#### Batched prompt inputs

```python
import cv2
import torch
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
input_boxes = torch.tensor([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
], device=predictor.device)
transformed_boxes = predictor.transform.applyboxestorch(input_boxes, image.shape[:2])
# inference
masks, _, _ = predictor.predicttorch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    showmask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    showbox(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()
```

#### End-to-end batched inference

```python
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.transforms import ResizeLongestSide

'''prepareimage'''
def prepareimage(image, transform, device):
    image = transform.applyimage(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

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

# predictor could be SAMPredictor(use_default_sam_h=True) or SAMPredictor(use_default_sam_l=True) or SAMPredictor(use_default_sam_b=True)
predictor = SAMPredictor(use_default_sam_h=True)
sam = predictor.model
# resize_transform
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
# read image
image1 = cv2.imread('images/truck.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread('images/groceries.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# set prompt
image1_boxes = torch.tensor([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
], device=sam.device)
image2_boxes = torch.tensor([
    [450, 170, 520, 350],
    [350, 190, 450, 350],
    [500, 170, 580, 350],
    [580, 170, 640, 350],
], device=sam.device)
# set batched_input
batched_input = [
     {
         'image': prepareimage(image1, resize_transform, sam),
         'boxes': resize_transform.applyboxestorch(image1_boxes, image1.shape[:2]),
         'original_size': image1.shape[:2]
     },
     {
         'image': prepareimage(image2, resize_transform, sam),
         'boxes': resize_transform.applyboxestorch(image2_boxes, image2.shape[:2]),
         'original_size': image2.shape[:2]
     }
]
# inference
batched_output = sam.inference(batched_input, multimask_output=False)
# show results
fig, ax = plt.subplots(1, 2, figsize=(20, 20))
ax[0].imshow(image1)
for mask in batched_output[0]['masks']:
    showmask(mask.cpu().numpy(), ax[0], random_color=True)
for box in image1_boxes:
    showbox(box.cpu().numpy(), ax[0])
ax[0].axis('off')
ax[1].imshow(image2)
for mask in batched_output[1]['masks']:
    showmask(mask.cpu().numpy(), ax[1], random_color=True)
for box in image2_boxes:
    showbox(box.cpu().numpy(), ax[1])
ax[1].axis('off')
plt.tight_layout()
plt.show()
```