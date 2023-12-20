## Introduction

<a href="https://github.com/chongzhou96/EdgeSAM">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/edgesam/edgesam.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2312.06660.pdf">EdgeSAM (ArXiv'2023)</a></summary>

```latex
@article{zhou2023edgesam,
    title={EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM},
    author={Zhou, Chong and Li, Xiangtai and Loy, Chen Change and Dai, Bo},
    journal={arXiv preprint arXiv:2312.06660},
    year={2023}
}
```

</details>


## Inference with EdgeSAM

### Object masks from prompts with EdgeSAM

#### Environment Set-up

Install sssegmentation:

```sh
# from pypi
pip install SSSegmentation
# from Github repository
pip install git+https://github.com/CharlesPikachu/sssegmentation.git
```

Download images:

```sh
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg
```

Refer to [EdgeSAM official repo](https://github.com/chongzhou96/EdgeSAM/blob/master/notebooks/predictor_example.ipynb), we provide some examples to use sssegmenation to generate object masks from prompts with EdgeSAM.

#### Selecting objects with EdgeSAM

To select the truck, choose a point on it. Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). 
Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
# set image
predictor.setimage(image)
# set prompt
input_label = np.array([1])
input_point = np.array([[500, 375]])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, num_multimask_outputs=4, use_stability_score=True
)
# show results
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    showmask(mask, plt.gca())
    showpoints(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f'mask_{i}.png')
```

#### Specifying a specific object with additional points

The single input point is ambiguous, and the model has returned multiple objects consistent with it. 
To obtain a single object, multiple points can be provided. 
If available, a mask from a previous iteration can also be supplied to the model to aid in prediction. 
When specifying a single object with multiple prompts, a single mask can be requested by setting `num_multimask_outputs=1`.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
# set image
predictor.setimage(image)
# set prompt
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, num_multimask_outputs=1
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

To exclude the car and specify just the window, a background point (with label 0, here shown in red) can be supplied.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
# set image
predictor.setimage(image)
# set prompt
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 0])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, num_multimask_outputs=1
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

#### Specifying a specific object with a box

The model can also take a box as input, provided in xyxy format.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
# set image
predictor.setimage(image)
# set prompt
input_box = np.array([425, 600, 700, 875])
# inference
masks, scores, logits = predictor.predict(
    point_coords=None, point_labels=None, box=input_box[None, :], num_multimask_outputs=1
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks, plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

#### Combining points and boxes

Points and boxes may be combined, just by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
# set image
predictor.setimage(image)
# set prompt
input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, box=input_box, num_multimask_outputs=1
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks, plt.gca())
showbox(input_box, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

#### Batched prompt inputs

`SAMPredictor` can take multiple input prompts for the same image, using `predicttorch` method. This method assumes input points are already torch tensors and have already been transformed to the input frame.

```python
import cv2
import torch
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
# set image
predictor.setimage(image)
# set prompt
input_boxes = torch.tensor([[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750],], device=predictor.device)
transformed_boxes = predictor.transform.applyboxestorch(input_boxes, image.shape[:2])
# inference
masks, scores, logits = predictor.predicttorch(
    point_coords=None, point_labels=None, boxes=transformed_boxes, num_multimask_outputs=1
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for i, (mask, score) in enumerate(zip(masks, scores)):
    showmask(mask.cpu().numpy(), plt.gca(), random_color=True)
    showbox(input_boxes[i].cpu().numpy(), plt.gca())
    plt.axis('off')
plt.savefig(f'mask.png')
```

#### End-to-end batched inference

If all prompts are available in advance, it is possible to run SAM directly in an end-to-end fashion. This also allows batching over images.

Both images and prompts are input as PyTorch tensors that are already transformed to the correct frame. Inputs are packaged as a list over images, which each element is a dict that takes the following keys:

- `image`: The input image as a PyTorch tensor in CHW format.
- `original_size`: The size of the image before transforming for input to SAM, in (H, W) format.
- `point_coords`: Batched coordinates of point prompts.
- `point_labels`: Batched labels of point prompts.
- `boxes`: Batched input boxes.
- `mask_inputs`: Batched input masks.

If a prompt is not present, the key can be excluded.

```python
import cv2
import torch
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.edgesam import EdgeSAMPredictor
from ssseg.modules.models.segmentors.sam.transforms import ResizeLongestSide
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

'''prepareimage'''
def prepareimage(image, transform, device):
    image = transform.applyimage(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

# predictor could be EdgeSAMPredictor(use_default_edgesam=True) or EdgeSAMPredictor(use_default_edgesam_3x=True)
predictor = EdgeSAMPredictor(use_default_edgesam=True, device='cpu')
edge_sam = predictor.model
# resize_transform
resize_transform = ResizeLongestSide(edge_sam.image_encoder.img_size)
# read image
image1 = cv2.imread('images/truck.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread('images/groceries.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# set prompt
image1_boxes = torch.tensor([
    [75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750],
], device=edge_sam.device)
image2_boxes = torch.tensor([
    [450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350],
], device=edge_sam.device)
# set batched_input
batched_input = [
    {
        'image': prepareimage(image1, resize_transform, edge_sam),
        'boxes': resize_transform.applyboxestorch(image1_boxes, image1.shape[:2]),
        'original_size': image1.shape[:2]
    },
    {
        'image': prepareimage(image2, resize_transform, edge_sam),
        'boxes': resize_transform.applyboxestorch(image2_boxes, image2.shape[:2]),
        'original_size': image2.shape[:2]
    }
]
# inference
batched_output = edge_sam.inference(batched_input, num_multimask_outputs=1)
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
plt.savefig(f'mask.png')
```

### Automatically generating object masks with EdgeSAM

The usage of `EdgeSAMAutomaticMaskGenerator` in EdgeSAM is exactly the same as SAM by replacing,

- `SAMAutomaticMaskGenerator`: `EdgeSAMAutomaticMaskGenerator`.

Specifically, you can import the class by

```python
from ssseg.modules.models.segmentors.edgesam import EdgeSAMAutomaticMaskGenerator

# mask_generator could be EdgeSAMAutomaticMaskGenerator(use_default_edgesam=True, device='cuda') or EdgeSAMAutomaticMaskGenerator(use_default_edgesam_3x=True, device='cuda')
mask_generator = EdgeSAMAutomaticMaskGenerator(use_default_edgesam=True, device='cuda')
```

By the way, you can refer to [inference-with-sam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-sam) to learn about how to use SAM with sssegmenation.
Also, you can refer to [EdgeSAM Official Repo](https://github.com/chongzhou96/EdgeSAM) to compare our implemented EdgeSAM with official version.