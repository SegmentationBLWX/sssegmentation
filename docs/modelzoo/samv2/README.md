## Introduction

<a href="https://github.com/facebookresearch/segment-anything-2">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/samv2/samv2.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2408.00714.pdf">SAMV2 (ArXiv'2024)</a></summary>

```latex
@article{ravi2024sam,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

</details>


## Inference with SAMV2

### Object masks in images from prompts with SAMV2

Segment Anything Model 2 (SAMV2) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt.

The `SAMV2ImagePredictor` class provides an easy interface to the model for prompting the model. It allows the user to first set an image using the `setimage` method, which calculates the necessary image embeddings. Then, prompts can be provided via the `predict` method to efficiently predict masks from those prompts. The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction.

#### Environment Set-up

To use SAMV2 in sssegmenation, `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1` are required. 
After installing the correct versions of python and torch components, you can install sssegmenation with SAMV2 on a GPU machine using the following commands:

```sh
git clone https://github.com/SegmentationBLWX/sssegmentation
cd sssegmentation
export COMPILE_SAMV2=1
python setup.py develop
```

Download images:

```sh
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/groceries.jpg
```

Refer to [SAMV2 official repo](https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb), we provide some examples to use sssegmenation to generate object masks from prompts with SAMV2.

#### Selecting objects with SAMV2

To select the truck, choose a point on it. Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.

```python
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
```

You can also access the example code from [examples/samv2/image/selectingobjectswithsamv2.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/selectingobjectswithsamv2.py).

#### Specifying a specific object with additional points

The single input point is ambiguous, and the model has returned multiple objects consistent with it. 
To obtain a single object, multiple points can be provided. 
If available, a mask from a previous iteration can also be supplied to the model to aid in prediction. 
When specifying a single object with multiple prompts, a single mask can be requested by setting `multimask_output=False`.

```python
'''
Function:
    SAMV2 examples: Specifying a specific object with additional points
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
# set prompt for the second time
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])
# inference for the second time
mask_input = logits[np.argmax(scores), :, :]
masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, mask_input=mask_input[None, :, :], multimask_output=False)
# show results
showmasks(image, masks, scores, point_coords=input_point, input_labels=input_label)
```

You can also access the example code from [examples/samv2/image/specifyingaspecificobjectwithadditionalpoints1.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/specifyingaspecificobjectwithadditionalpoints1.py).

To exclude the car and specify just the window, a background point (with label 0, here shown in red) can be supplied.

```python
'''
Function:
    SAMV2 examples: Specifying a specific object with additional points
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
# set prompt for the second time
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 0])
# inference for the second time
mask_input = logits[np.argmax(scores), :, :]
masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, mask_input=mask_input[None, :, :], multimask_output=False)
# show results
showmasks(image, masks, scores, point_coords=input_point, input_labels=input_label)
```

You can also access the example code from [examples/samv2/image/specifyingaspecificobjectwithadditionalpoints2.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/specifyingaspecificobjectwithadditionalpoints2.py).

#### Specifying a specific object with a box

The model can also take a box as input, provided in xyxy format.

```python
'''
Function:
    SAMV2 examples: Specifying a specific object with a box
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
input_box = np.array([425, 600, 700, 875])
# inference
masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
# show results
showmasks(image, masks, scores, box_coords=input_box)
```

You can also access the example code from [examples/samv2/image/specifyingaspecificobjectwithabox.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/specifyingaspecificobjectwithabox.py).

#### Combining points and boxes

Points and boxes may be combined, just by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

```python
'''
Function:
    SAMV2 examples: Combining points and boxes
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
input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])
# inference
masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False)
# show results
showmasks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)
```

You can also access the example code from [examples/samv2/image/combiningpointsandboxes.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/combiningpointsandboxes.py).

#### Batched prompt inputs

`SAMV2ImagePredictor` can take multiple input prompts for the same image, using predict method. For example, imagine we have several box outputs from an object detector.

```python
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
```

You can also access the example code from [examples/samv2/image/batchedpromptinputs.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/batchedpromptinputs.py).

#### End-to-end batched inference

If all prompts are available in advance, it is possible to run SAMV2 directly in an end-to-end fashion. This also allows batching over images.

```python
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
```

You can also access the example code from [examples/samv2/image/endtoendbatchedinference1.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/endtoendbatchedinference1.py).

Similarly, we can have a batch of point prompts defined over a batch of images.

```python
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
```

You can also access the example code from [examples/samv2/image/endtoendbatchedinference2.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/endtoendbatchedinference2.py).

### Automatically generating object masks with SAMV2

Since SAMV2 can efficiently process prompts, masks for the entire image can be generated by sampling a large number of prompts over an image.

The class `SAMV2AutomaticMaskGenerator` implements this capability. 
It works by sampling single-point input prompts in a grid over the image, from each of which SAM can predict multiple masks. 
Then, masks are filtered for quality and deduplicated using non-maximal suppression. 
Additional options allow for further improvement of mask quality and quantity, such as running prediction on multiple crops of the image or postprocessing masks to remove small disconnected regions and holes.

#### Environment Set-up

To use SAMV2 in sssegmenation, `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1` are required. 
After installing the correct versions of python and torch components, you can install sssegmenation with SAMV2 on a GPU machine using the following commands:

```sh
git clone https://github.com/SegmentationBLWX/sssegmentation
cd sssegmentation
export COMPILE_SAMV2=1
python setup.py develop
```

Download images:

```sh
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/cars.jpg
```

Refer to [SAMV2 official repo](https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb), we provide some examples to use sssegmenation to automatically generate object masks with SAMV2.

#### Automatic mask generation

To generate masks, just run `generate` on an image after instancing `SAMV2AutomaticMaskGenerator`.

```python
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
```

You can also access the example code from [examples/samv2/image/automaticmaskgeneration.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/automaticmaskgeneration.py).

Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

- `segmentation`: the mask,
- `area`: the area of the mask in pixels,
- `bbox`: the boundary box of the mask in XYWH format,
- `predicted_iou`: the model's own prediction for the quality of the mask,
- `point_coords`: the sampled input point that generated this mask,
- `stability_score`: an additional measure of mask quality,
- `crop_box`: the crop of the image used to generate this mask in XYWH format.

#### Automatic mask generation options

There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. 
Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. 
Here is an example configuration that samples more masks:

```python
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
```

You can also access the example code from [examples/samv2/image/automaticmaskgenerationoptions.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/image/automaticmaskgenerationoptions.py).

### Video segmentation with SAMV2

This section shows how to use SAMV2 for interactive segmentation in videos. It will cover the following:

- adding clicks on a frame to get and refine *masklets* (spatio-temporal masks),
- propagating clicks to get *masklets* throughout the video,
- segmenting and tracking multiple objects at the same time.

We use the terms *segment* or *mask* to refer to the model prediction for an object on a single frame, and *masklet* to refer to the spatio-temporal masks across the entire video.

#### Environment Set-up

To use SAMV2 in sssegmenation, `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1` are required. 
After installing the correct versions of python and torch components, you can install sssegmenation with SAMV2 on a GPU machine using the following commands:

```sh
git clone https://github.com/SegmentationBLWX/sssegmentation
cd sssegmentation
export COMPILE_SAMV2=1
python setup.py develop
```

Download video:

```sh
wget -P videos https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_sam2/bedroom.zip
cd videos
unzip bedroom.zip
cd ..
```

Here, we assume that the video is stored as a list of JPEG frames with filenames like `<frame_index>.jpg`.

For your custom videos, you can extract their JPEG frames using [ffmpeg](https://ffmpeg.org/) as follows:

```sh
ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
```

where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks ffmpeg to start the JPEG file from `00000.jpg`.

Refer to [SAMV2 official repo](https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb), we provide some examples to use sssegmenation to perform video segmentation with SAMV2.

#### Segment & track one object

**Step1: Add a first click on a frame**

```python
'''
Function:
    SAMV2 examples: Segment & track one object
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Add a first click on a frame
# To get started, let's try to segment the child on the left.
# Here we make a positive click at (x, y) = (210, 350) with label `1`, by sending their coordinates and labels into the `addnewpoints` API.
# Note: label `1` indicates a positive click (to add a region) while label `0` indicates a negative click (to remove a region).
# the frame index we interact with
ann_frame_idx = 0
# give a unique id to each object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[210, 350]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
showpoints(points, labels, plt.gca())
showmask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.savefig('output_step1.png')
```

You can also access the example code from [examples/samv2/video/segmenttrackoneobject_step1.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmenttrackoneobject_step1.py).

**Step2: Add a second click to refine the prediction**

Hmm, it seems that although we wanted to segment the child on the left, the model predicts the mask for only the shorts -- this can happen since there is ambiguity from a single click about what the target object should be. 
We can refine the mask on this frame via another positive click on the child's shirt.

Here we make a second positive click at `(x, y) = (250, 220)` with label `1` to expand the mask.
(Note: we need to send all the clicks and their labels (i.e. not just the last click) when calling `addnewpoints`.)

```python
'''
Function:
    SAMV2 examples: Segment & track one object
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Add a first click on a frame
# To get started, let's try to segment the child on the left.
# Here we make a positive click at (x, y) = (210, 350) with label `1`, by sending their coordinates and labels into the `addnewpoints` API.
# Note: label `1` indicates a positive click (to add a region) while label `0` indicates a negative click (to remove a region).
# the frame index we interact with
ann_frame_idx = 0
# give a unique id to each object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask, sending all clicks (and their labels) to `addnewpoints`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
showpoints(points, labels, plt.gca())
showmask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.savefig('output_step2.png')
```

You can also access the example code from [examples/samv2/video/segmenttrackoneobject_step2.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmenttrackoneobject_step2.py).

With this 2nd refinement click, now we get a segmentation mask of the entire child on frame 0.

**Step 3: Propagate the prompts to get the masklet across the video**

To get the masklet throughout the entire video, we propagate the prompts using the `propagateinvideo` API.

```python
'''
Function:
    SAMV2 examples: Segment & track one object
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Add a first click on a frame
# To get started, let's try to segment the child on the left.
# Here we make a positive click at (x, y) = (210, 350) with label `1`, by sending their coordinates and labels into the `addnewpoints` API.
# Note: label `1` indicates a positive click (to add a region) while label `0` indicates a negative click (to remove a region).
# the frame index we interact with
ann_frame_idx = 0
# give a unique id to each object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask, sending all clicks (and their labels) to `addnewpoints`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# render the segmentation results every few frames
vis_frame_stride = 15
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        showmask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(f'out_frame_{out_frame_idx}.png')
    plt.cla()
    plt.clf()
```

You can also access the example code from [examples/samv2/video/segmenttrackoneobject_step3.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmenttrackoneobject_step3.py).

**Step 4: Add new prompts to further refine the masklet**

It appears that in the output masklet above, there are some imperfections in boundary details on frame 150.

With SAMV2 we can fix the model predictions interactively. 
We can add a negative click at `(x, y) = (82, 415)` on this frame with label `0` to refine the masklet. 
Here we call the `addnewpoints` API with a different `frame_idx` argument to indicate the frame index we want to refine.

```python
'''
Function:
    SAMV2 examples: Segment & track one object
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Add a first click on a frame
# To get started, let's try to segment the child on the left.
# Here we make a positive click at (x, y) = (210, 350) with label `1`, by sending their coordinates and labels into the `addnewpoints` API.
# Note: label `1` indicates a positive click (to add a region) while label `0` indicates a negative click (to remove a region).
# the frame index we interact with
ann_frame_idx = 0
# give a unique id to each object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask, sending all clicks (and their labels) to `addnewpoints`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# further refine some details on this frame
ann_frame_idx = 150
# give a unique id to the object we interact with (it can be any integers)
ann_obj_id = 1
# show the segment before further refinement
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx} -- before refinement")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
showmask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)
plt.savefig(f"frame {ann_frame_idx} -- before refinement.png")
plt.cla()
plt.clf()
# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
points = np.array([[82, 415]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([0], np.int32)
_, _, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# show the segment after the further refinement
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx} -- after refinement")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
showpoints(points, labels, plt.gca())
showmask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)
plt.savefig(f"frame {ann_frame_idx} -- after refinement.png")
plt.cla()
plt.clf()
```

You can also access the example code from [examples/samv2/video/segmenttrackoneobject_step4.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmenttrackoneobject_step4.py).

**Step 5: Propagate the prompts (again) to get the masklet across the video**

Let's get an updated masklet for the entire video. Here we call `propagateinvideo` again to propagate all the prompts after adding the new refinement click above.

```python
'''
Function:
    SAMV2 examples: Segment & track one object
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Add a first click on a frame
# To get started, let's try to segment the child on the left.
# Here we make a positive click at (x, y) = (210, 350) with label `1`, by sending their coordinates and labels into the `addnewpoints` API.
# Note: label `1` indicates a positive click (to add a region) while label `0` indicates a negative click (to remove a region).
# the frame index we interact with
ann_frame_idx = 0
# give a unique id to each object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask, sending all clicks (and their labels) to `addnewpoints`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# further refine some details on this frame
ann_frame_idx = 150
# give a unique id to the object we interact with (it can be any integers)
ann_obj_id = 1
# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
points = np.array([[82, 415]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([0], np.int32)
_, _, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# render the segmentation results every few frames
vis_frame_stride = 15
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        showmask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(f'out_frame_{out_frame_idx}.png')
    plt.cla()
    plt.clf()
```

You can also access the example code from [examples/samv2/video/segmenttrackoneobject_step5.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmenttrackoneobject_step5.py).

The segments now look good on all frames.

#### Segment multiple objects simultaneously

**Step 1: Add two objects on a frame**

SAMV2 can also segment and track two or more objects at the same time. 
One way, of course, is to do them one by one. 
However, it would be more efficient to batch them together (e.g. so that we can share the image features between objects to reduce computation costs).

This time, let's focus on object parts and segment the shirts of both childen in this video. 

```python
'''
Function:
    SAMV2 examples: Segment multiple objects simultaneously
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Here we add prompts for these two objects and assign each of them a unique object id. (hold all the clicks we add for visualization)
prompts = {}
# Add the first object (the left child's shirt) with a positive click at (x, y) = (200, 300) and a negative click at (x, y) = (275, 175) on frame 0.
# We assign it to object id 2 (it can be arbitrary integers, and only needs to be unique for each object to track), which is passed to the `addnewpoints` API to distinguish the object we are clicking upon.
ann_frame_idx = 0
ann_obj_id = 2
# Let's add a positive click at (x, y) = (200, 300) and a negative click at (x, y) = (275, 175) to get started on the first object
points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
# save to prompts
prompts[ann_obj_id] = points, labels
# sending all clicks (and their labels) to `addnewpoints`
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# Let's move on to the second object (the right child's shirt) with a positive click at (x, y) = (400, 150) on frame 0. 
# Here we assign object id 3 to this second object (it can be arbitrary integers, and only needs to be unique for each object to track).
# Note: when there are multiple objects, the `addnewpoints` API will return a list of masks for each object.
ann_frame_idx = 0
ann_obj_id = 3
# Let's now move on to the second object we want to track (giving it object id `3`) with a positive click at (x, y) = (400, 150)
points = np.array([[400, 150]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
# save to prompts
prompts[ann_obj_id] = points, labels
# `addnewpoints` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
showpoints(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    showpoints(*prompts[out_obj_id], plt.gca())
    showmask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig('output.png')
```

You can also access the example code from [examples/samv2/video/segmentmultipleobjectssimultaneously_step1.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmentmultipleobjectssimultaneously_step1.py).

**Step 2: Propagate the prompts to get masklets across the video**

Now, we propagate the prompts for both objects to get their masklets throughout the video.

Note: when there are multiple objects, the `propagateinvideo` API will return a list of masks for each object.

```python
'''
Function:
    SAMV2 examples: Segment multiple objects simultaneously
Author:
    Zhenchao Jin
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssseg.modules.models.segmentors.samv2 import SAMV2VideoPredictor
from ssseg.modules.models.segmentors.samv2.visualization import showpoints


'''showmask'''
def showmask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# initialize environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# pre-load video
video_dir = "./videos/bedroom"
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# predictor could be SAMV2VideoPredictor(use_default_samv2_t=True) or SAMV2VideoPredictor(use_default_samv2_s=True) or SAMV2VideoPredictor(use_default_samv2_bplus=True) or SAMV2VideoPredictor(use_default_samv2_l=True)
predictor = SAMV2VideoPredictor(use_default_samv2_l=True, device='cuda')
# Initialize the inference state
# SAMV2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.initstate(video_path=video_dir)
# Note: if you have run any previous tracking using this `inference_state`, please reset it first via `resetstate`.
predictor.resetstate(inference_state)
# Here we add prompts for these two objects and assign each of them a unique object id. (hold all the clicks we add for visualization)
prompts = {}
# Add the first object (the left child's shirt) with a positive click at (x, y) = (200, 300) and a negative click at (x, y) = (275, 175) on frame 0.
# We assign it to object id 2 (it can be arbitrary integers, and only needs to be unique for each object to track), which is passed to the `addnewpoints` API to distinguish the object we are clicking upon.
ann_frame_idx = 0
ann_obj_id = 2
# Let's add a positive click at (x, y) = (200, 300) and a negative click at (x, y) = (275, 175) to get started on the first object
points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
# save to prompts
prompts[ann_obj_id] = points, labels
# sending all clicks (and their labels) to `addnewpoints`
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# Let's move on to the second object (the right child's shirt) with a positive click at (x, y) = (400, 150) on frame 0. 
# Here we assign object id 3 to this second object (it can be arbitrary integers, and only needs to be unique for each object to track).
# Note: when there are multiple objects, the `addnewpoints` API will return a list of masks for each object.
ann_frame_idx = 0
ann_obj_id = 3
# Let's now move on to the second object we want to track (giving it object id `3`) with a positive click at (x, y) = (400, 150)
points = np.array([[400, 150]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
# save to prompts
prompts[ann_obj_id] = points, labels
# `addnewpoints` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.addnewpoints(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
# run propagation throughout the video and collect the results in a dict (video_segments contains the per-frame segmentation results)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagateinvideo(inference_state):
    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
# render the segmentation results every few frames
vis_frame_stride = 15
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        showmask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(f'out_frame_{out_frame_idx}.png')
    plt.cla()
    plt.clf()
```

You can also access the example code from [examples/samv2/video/segmentmultipleobjectssimultaneously_step2.py](https://github.com/SegmentationBLWX/sssegmentation/blob/main/examples/samv2/video/segmentmultipleobjectssimultaneously_step2.py).

Looks like both children's shirts are well segmented in this video.

Now you can try SAMV2 on your own videos and use cases!