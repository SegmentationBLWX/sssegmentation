# Get Started


## Prerequisites

In this section, we introduce some prerequisites for using SSSegmentation. 
If you are experienced with Python and PyTorch and have already installed them, just skip this part.

**1.Operation System**

SSSegmentation works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.3+.

**2.Anaconda**

For Linux and Mac users, we strongly recommend you to download and install [Anaconda](https://docs.conda.io/en/latest/miniconda.html).
Then, you can create a conda environment for SSSegmentation and activate it,

```sh
conda create --name ssseg python=3.8 -y
conda activate ssseg
```

**3.WGET & Decompression Software**

If you want to utilize the provided scripts to prepare the datasets, it is necessary for you to install wget (for downloading datasets), 7z (for processing compressed packages) and tar (for processing compressed packages) in your operation system.
For windows users, the resources are listed as following,

- 7Z: [Download](https://sparanoid.com/lab/7z/download.html),
- RAR: [Download](https://www.win-rar.com/start.html?&L=0),
- WGET: [Download](http://downloads.sourceforge.net/gnuwin32/wget-1.11.4-1-setup.exe?spm=a2c6h.12873639.article-detail.7.3f825677H6sKF2&file=wget-1.11.4-1-setup.exe).

Besides, [Cmder](https://cmder.app/) are recommended to help the windows users execute the provided scripts successfully.

## Installation

**1.Clone SSSegmentation**

You can run the following commands to clone the sssegmentation repository,

```sh 
git clone https://github.com/SegmentationBLWX/sssegmentation.git
cd sssegmentation
```

**2.Install Requirements**

**2.1 Basic Requirements (Necessary)**

To set up the essential prerequisites for running SSSegmentation, execute the following commands,

```sh
pip install -r requirements.txt
```

This command will automatically install the following packages,

- `chainercv`: set in requirements/evaluate.txt,
- `cityscapesscripts`: set in requirements/evaluate.txt,
- `pycocotools`: set in requirements/evaluate.txt,
- `pillow`: set in requirements/io.txt,
- `pandas`: set in requirements/io.txt,
- `opencv-python`: set in requirements/io.txt,
- `numpy`: set in requirements/science.txt,
- `scipy`: set in requirements/science.txt,
- `tqdm`: set in requirements/terminal.txt,
- `argparse`: set in requirements/terminal.txt,
- `cython`: set in requirements/misc.txt.

**2.2 Pytorch and Torchvision (Necessary)**

If you intend to utilize SSSegmentation, it is imperative to install PyTorch and torchvision. 
We recommend you to follow the [official instructions](https://pytorch.org/get-started/previous-versions/) to install them, *e.g.*,

```sh
# CUDA 11.0
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.2
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```

**2.3 MMCV (Optional)**

If you want to install mmcv-full for training some segmentors with mmcv-integrated operators like [CCNet](https://arxiv.org/pdf/1811.11721.pdf) and [Mask2Former](https://arxiv.org/pdf/2112.01527.pdf), we recommend you to install the pre-build mmcv-full package as below,

```sh
# mmcv < 2.0.0, mmcv versions include mmcv-full and mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# mmcv > 2.0.0, mmcv versions include mmcv and mmcv-lite
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

For more details, you can refer to [MMCV official repository](https://github.com/open-mmlab/mmcv).

**2.4 Apex (Optional)**

If you want to install apex for [Mixed Precision (FP16) Training](https://arxiv.org/pdf/1710.03740.pdf), we recommend you to follow the instructions in [official repository](https://github.com/NVIDIA/apex).
For example, the Linux users can install apex with the following commands,

```sh
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# a Python-only build
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```

Then, you need to turn on mixed precision training in corresponding config file as follow,

```python
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'apex', 'initialize': {'opt_level': 'O1'}, 'scale_loss': {}}
```

Note that, SSSegmentation supports two types of mixed precision training, *i.e.*, 'apex' and 'pytorch'.
If you want to use mixed precision training supported by [Pytorch](https://pytorch.org/), you only need to install torch with `torch.__version >= 1.5.0`, *e.g.*,

```sh
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

Then, you need to turn on mixed precision training in corresponding config file as follow,

```python
import torch

SEGMENTOR_CFG['fp16_cfg'] = {'type': 'pytorch', 'autocast': {'dtype': torch.float16}, 'grad_scaler': {}}
```

**2.5 TIMM (Optional)**

SSSegmentation provides support for importing backbone networks from [timm](https://github.com/huggingface/pytorch-image-models) to train our segmentors. To install timm, you can simply run,

```sh
pip install timm
```

For more details, you can refer to [TIMM official repository](https://github.com/huggingface/pytorch-image-models) and [SSSegmentation timm backbone wrapper](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/timmwrapper.py).

**2.6 Albumentations (Optional)**

SSSegmentation provides support for importing data augmentation transforms from [albumentations](https://github.com/albumentations-team/albumentations) to train our segmentors. To install albumentations, you can simply run,

```sh
pip install -U albumentations
```

For more details, you can refer to [Albumentations official repository](https://github.com/albumentations-team/albumentations) and [SSSegmentation albumentations wrapper](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py).