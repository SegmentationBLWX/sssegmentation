# Advanced API


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

Refer to [SAM official repo](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb), we provide some examples to use sssegmenation to generate object masks from prompts with SAM.

#### Selecting objects with SAM

To select the truck, choose a point on it. Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). 
Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

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
    point_coords=input_point, point_labels=input_label, multimask_output=True,
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
When specifying a single object with multiple prompts, a single mask can be requested by setting `multimask_output=False`.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

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
    point_coords=input_point, point_labels=input_label, multimask_output=True,
)
# set prompt for the second time
input_label = np.array([1, 1])
input_point = np.array([[500, 375], [1125, 625]])
# inference for the second time
mask_input = logits[np.argmax(scores), :, :]
masks, _, _ = predictor.predict(
    point_coords=input_point, point_labels=input_label, mask_input=mask_input[None, :, :], multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

To exclude the car and specify just the window, a background point (with label 0, here shown in red) can be supplied.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

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
    point_coords=input_point, point_labels=input_label, multimask_output=True,
)
# set prompt for the second time
input_label = np.array([1, 0])
input_point = np.array([[500, 375], [1125, 625]])
# inference for the second time
mask_input = logits[np.argmax(scores), :, :]
masks, _, _ = predictor.predict(
    point_coords=input_point, point_labels=input_label, mask_input=mask_input[None, :, :], multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

#### Specifying a specific object with a box

The model can also take a box as input, provided in xyxy format.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

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
    point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

#### Combining points and boxes

Points and boxes may be combined, just by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

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
    point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

#### Batched prompt inputs

`SAMPredictor` can take multiple input prompts for the same image, using `predicttorch` method. This method assumes input points are already torch tensors and have already been transformed to the input frame.

```python
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMPredictor(use_default_sam_h=True) or SAMPredictor(use_default_sam_l=True) or SAMPredictor(use_default_sam_b=True)
predictor = SAMPredictor(use_default_sam_h=True)
# set image
predictor.setimage(image)
# set prompt
input_boxes = torch.tensor([
    [75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750],
], device=predictor.device)
transformed_boxes = predictor.transform.applyboxestorch(input_boxes, image.shape[:2])
# inference
masks, _, _ = predictor.predicttorch(
    point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    showmask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    showbox(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.savefig('mask.png')
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
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.sam import SAMPredictor
from ssseg.modules.models.segmentors.sam.transforms import ResizeLongestSide
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

'''prepareimage'''
def prepareimage(image, transform, device):
    image = transform.applyimage(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

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
    [75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750],
], device=sam.device)
image2_boxes = torch.tensor([
    [450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350],
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
plt.savefig('mask.png')
```

The output is a list over results for each input image, where list elements are dictionaries with the following keys:

- `masks`: A batched torch tensor of predicted binary masks, the size of the original image.
- `iou_predictions`: The model's prediction of the quality for each mask.
- `low_res_logits`: Low res logits for each mask, which can be passed back to the model as mask input on a later iteration.

### Automatically generating object masks with SAM

Since SAM can efficiently process prompts, masks for the entire image can be generated by sampling a large number of prompts over an image. This method was used to generate the dataset SA-1B. 

The class `SAMAutomaticMaskGenerator` implements this capability. 
It works by sampling single-point input prompts in a grid over the image, from each of which SAM can predict multiple masks. 
Then, masks are filtered for quality and deduplicated using non-maximal suppression. 
Additional options allow for further improvement of mask quality and quantity, such as running prediction on multiple crops of the image or postprocessing masks to remove small disconnected regions and holes.

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
from ssseg.modules.models.segmentors.sam.visualization import showanns
from ssseg.modules.models.segmentors.sam import SAMAutomaticMaskGenerator

# read image
image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# mask generator
mask_generator = SAMAutomaticMaskGenerator(use_default_sam_h=True, device='cuda')
# generate masks on an image
masks = mask_generator.generate(image)
# show all the masks overlayed on the image
plt.figure(figsize=(20, 20))
plt.imshow(image)
showanns(masks)
plt.axis('off')
plt.savefig('mask.png')
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
from ssseg.modules.models.segmentors.sam.visualization import showanns
from ssseg.modules.models.segmentors.sam import SAMAutomaticMaskGenerator

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
plt.figure(figsize=(20, 20))
plt.imshow(image)
showanns(masks)
plt.axis('off')
plt.savefig('mask.png')
```


## Inference with MobileSAM

The usage of MobileSAM in sssegmenation is exactly the same as SAM by replacing

- `SAM`: `MobileSAM`,
- `SAMPredictor`: `MobileSAMPredictor`,
- `SAMAutomaticMaskGenerator`: `MobileSAMAutomaticMaskGenerator`.

Specifically, you can import the three classes by

```python
from ssseg.modules.models.segmentors.mobilesam import MobileSAM
from ssseg.modules.models.segmentors.mobilesam import MobileSAMPredictor
from ssseg.modules.models.segmentors.mobilesam import MobileSAMAutomaticMaskGenerator

# predictor only could be MobileSAMPredictor(use_default_sam_t_5m=True, device='cuda')
predictor = MobileSAMPredictor(use_default_sam_t_5m=True, device='cuda')

# mask_generator only could be MobileSAMAutomaticMaskGenerator(use_default_sam_t_5m=True, device='cuda')
mask_generator = MobileSAMAutomaticMaskGenerator(use_default_sam_t_5m=True, device='cuda')
```

By the way, you can refer to [inference-with-sam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-sam) to learn about how to use SAM with sssegmenation.
Also, you can refer to [MobileSAM Official Repo](https://github.com/ChaoningZhang/MobileSAM) to compare our implemented MobileSAM with official version.


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


## Inference with SAMHQ

### Object masks from prompts with SAMHQ

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
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example0.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example1.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example2.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example3.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example4.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example5.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example6.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example7.png
wget -P images https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png
```

Refer to [SAMHQ official repo](https://colab.research.google.com/drive/1QwAbn5hsdqKOD5niuBzuqQX4eLCbNKFL?usp=sharing), we provide some examples to use sssegmenation to generate object masks from prompts with SAMHQ.

#### Specifying a specific object with a box

The model can take a box as input, provided in xyxy format. 

Here is an example that uses SAMHQ to select tennis rackets with a box as prompt and set `hq_token_only=False`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example0.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True)
# set image
predictor.setimage(image)
# set prompt
input_box = np.array([4, 13, 1007, 1023])
# inference
masks, scores, logits = predictor.predict(
    point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False, hq_token_only=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

Here is an example that uses SAMHQ to select a butterfly with a box as prompt and set `hq_token_only=True`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True)
# set image
predictor.setimage(image)
# set prompt
input_box = np.array([306, 132, 925, 893])
# inference
masks, scores, logits = predictor.predict(
    point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False, hq_token_only=True,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

Here is an example that uses SAMHQ to select a chair with a box as prompt and set `hq_token_only=True`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example4.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True)
# set image
predictor.setimage(image)
# set prompt
input_box = np.array([64, 76, 940, 919])
# inference
masks, scores, logits = predictor.predict(
    point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False, hq_token_only=True,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

Here is an example that uses SAMHQ to select a whale with a box as prompt and set `hq_token_only=False`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example6.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True)
# set image
predictor.setimage(image)
# set prompt
input_box = np.array([181, 196, 757, 495])
# inference
masks, scores, logits = predictor.predict(
    point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False, hq_token_only=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks[0], plt.gca())
showbox(input_box, plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

#### Specifying a specific object with points

To select a object, you can also choose a point or some points on it. Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). 

Here is an example that uses SAMHQ to select a chair with two points as prompt and set `hq_token_only=True`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True, device='cuda')
# set image
predictor.setimage(image)
# set prompt
input_point = np.array([[495, 518], [217, 140]])
input_label = np.array([1, 1])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, multimask_output=False, hq_token_only=True,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

Here is an example that uses SAMHQ to select a steel frame with three points as prompt and set `hq_token_only=False`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example3.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True, device='cuda')
# set image
predictor.setimage(image)
# set prompt
input_point = np.array([[221, 482], [498, 633], [750, 379]])
input_label = np.array([1, 1, 1])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, multimask_output=False, hq_token_only=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

Here is an example that uses SAMHQ to select an eagle with two points as prompt and set `hq_token_only=False`,

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example5.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True, device='cuda')
# set image
predictor.setimage(image)
# set prompt
input_point = np.array([[373, 363], [452, 575]])
input_label = np.array([1, 1])
# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point, point_labels=input_label, multimask_output=False, hq_token_only=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
showmask(masks, plt.gca())
showpoints(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig(f'mask.png')
```

#### Batched prompt inputs

`SAMPredictor` can take multiple input prompts for the same image, using `predicttorch` method. This method assumes input points are already torch tensors and have already been transformed to the input frame.

Here is an example that uses SAMHQ to select a bed and a chair with two boxes as prompt and set `hq_token_only=False`,

```python
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ssseg.modules.models.segmentors.samhq import SAMHQPredictor
from ssseg.modules.models.segmentors.sam.visualization import showmask, showpoints, showbox

# read image
image = cv2.imread('images/example7.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# predictor could be SAMHQPredictor(use_default_samhq_t_5m=True) or SAMHQPredictor(use_default_samhq_b=True) or SAMHQPredictor(use_default_samhq_l=True) or SAMHQPredictor(use_default_samhq_h=True)
predictor = SAMHQPredictor(use_default_samhq_l=True)
# set image
predictor.setimage(image)
# set prompt
input_boxes = torch.tensor([
    [45, 260, 515, 470], [310, 228, 424, 296]
], device=predictor.device)
transformed_boxes = predictor.transform.applyboxestorch(input_boxes, image.shape[:2])
# inference
masks, _, _ = predictor.predicttorch(
    point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False, hq_token_only=False,
)
# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    showmask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    showbox(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.savefig('mask.png')
```

### Automatically generating object masks with SAMHQ

The usage of `SAMHQAutomaticMaskGenerator` in SAMHQ is exactly the same as SAM by replacing,

- `SAMAutomaticMaskGenerator`: `SAMHQAutomaticMaskGenerator`.

Specifically, you can import the class by

```python
from ssseg.modules.models.segmentors.samhq import SAMHQAutomaticMaskGenerator

# mask_generator could be SAMHQAutomaticMaskGenerator(use_default_samhq_t_5m=True, device='cuda') or SAMHQAutomaticMaskGenerator(use_default_samhq_b=True, device='cuda') or SAMHQAutomaticMaskGenerator(use_default_samhq_l=True, device='cuda') or SAMHQAutomaticMaskGenerator(use_default_samhq_h=True, device='cuda')
mask_generator = SAMHQAutomaticMaskGenerator(use_default_samhq_l=True, device='cuda')
```

By the way, you can refer to [inference-with-sam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-sam) to learn about how to use SAM with sssegmenation.
Also, you can refer to [SAMHQ Official Repo](https://github.com/SysCV/sam-hq) to compare our implemented SAMHQ with official version.
