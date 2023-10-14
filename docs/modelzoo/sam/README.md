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

### Object masks from prompts with SAM

The Segment Anything Model (SAM) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt. 

The `SAMPredictor` class provides an easy interface to the model for prompting the model. 
It allows the user to first set an image using the `setimage` method, which calculates the necessary image embeddings. 
Then, prompts can be provided via the `predict` method to efficiently predict masks from those prompts. 
The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction.

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

Refer to [SAM official repo](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb), we provide some examples to use sssegmenation to generate object masks from prompts with SAM.

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

### Automatically generating object masks with SAM

Since SAM can efficiently process prompts, masks for the entire image can be generated by sampling a large number of prompts over an image. This method was used to generate the dataset SA-1B. 

The class `SAMAutomaticMaskGenerator` implements this capability. 
It works by sampling single-point input prompts in a grid over the image, from each of which SAM can predict multiple masks. 
Then, masks are filtered for quality and deduplicated using non-maximal suppression. 
Additional options allow for further improvement of mask quality and quantity, such as running prediction on multiple crops of the image or postprocessing masks to remove small disconnected regions and holes.

#### Environment Set-up

Install sssegmentation:

```sh
git clone https://github.com/SegmentationBLWX/sssegmentation
cd sssegmentation
pip install -r requirements.txt # install dependencies refer to https://sssegmentation.readthedocs.io/en/latest/GetStarted.html#prerequisites
```

Download images:

```sh
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg
```

Refer to [SAM official repo](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb), we provide some examples to use sssegmenation to automatically generating object masks with SAM.

#### Automatic mask generation

To run automatic mask generation, provide a SAM model to the `SAMAutomaticMaskGenerator` class. Set the path below to the SAM checkpoint. Running on CUDA and with the default model is recommended.

```python
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMAutomaticMaskGenerator

'''showanns'''
def showanns(anns):
    if len(anns) == 0: return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# read image
image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# mask generator
mask_generator = SAMAutomaticMaskGenerator(use_default_sam_h=True, device='cuda')
# generate masks on an image
masks = mask_generator.generate(image)
# show all the masks overlayed on the image
plt.figure(figsize=(20,20))
plt.imshow(image)
showanns(masks)
plt.axis('off')
plt.show()
```

Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

- `segmentation` : the mask,
- `area` : the area of the mask in pixels,
- `bbox` : the boundary box of the mask in XYWH format,
- `predicted_iou` : the model's own prediction for the quality of the mask,
- `point_coords` : the sampled input point that generated this mask,
- `stability_score` : an additional measure of mask quality,
- `crop_box` : the crop of the image used to generate this mask in XYWH format.

#### Automatic mask generation options

There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. 
Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. Here is an example configuration that samples more masks:

```python
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMAutomaticMaskGenerator

'''showanns'''
def showanns(anns):
    if len(anns) == 0: return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# read image
image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# mask generator
mask_generator = SAMAutomaticMaskGenerator(
    use_default_sam_h=True, device='cuda', points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92,
    crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100,
)
# generate masks on an image
masks = mask_generator.generate(image)
# show all the masks overlayed on the image
plt.figure(figsize=(20,20))
plt.imshow(image)
showanns(masks)
plt.axis('off')
plt.show()
```