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

Install sssegmentation:

```sh
# from pypi
pip install SSSegmentation
# from Github repository
pip install git+https://github.com/SegmentationBLWX/sssegmentation.git
# locally install
git clone https://github.com/SegmentationBLWX/sssegmentation
cd sssegmentation
pip install -e .
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

