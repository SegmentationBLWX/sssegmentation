## Introduction

<a href="https://github.com/SysCV/sam-hq">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/samhq/samhq.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2306.01567.pdf">SAMHQ (NeurIPS'2023)</a></summary>

```latex
@article{ke2023segment,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    journal={arXiv preprint arXiv:2306.01567},
    year={2023}
}
```

</details>


## Inference with SAMHQ

### Object masks from prompts with SAMHQ

#### Environment Set-up

Install sssegmentation:

```sh
# from pypi
pip install SSSegmentation
# from Github repository
pip install git+https://github.com/SegmentationBLWX/sssegmentation.git
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
