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
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg
```

Refer to [SAMV2 official repo](https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb), we provide some examples to use sssegmenation to generate object masks from prompts with SAMV2.

#### Selecting objects with SAMV2

To select the truck, choose a point on it. Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.

```python
```