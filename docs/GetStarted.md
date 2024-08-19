# Get Started

SSSegmentation is an open source supervised semantic segmentation toolbox based on pytorch.

In this chapter, we demonstrate some necessary preparations before developing or using SSSegmentation.


## Install SSSegmentation for Developing

SSSegmentation works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+. 

If you are experienced with Python and PyTorch and have already installed them, just skip this section and jump to the next section [Prepare Datasets](https://sssegmentation.readthedocs.io/en/latest/GetStarted.html#prepare-datasets).
Otherwise, you can follow the instructions in this section for installing SSSegmentation.

#### Install Anaconda

Anaconda is an open-source package and environment management system that runs on Windows, macOS, and Linux. 

We recommend the users to download and install Anaconda to create an independent environment for SSSegmentation.
Specifically, you can download and install Anaconda or Miniconda from the [official website](https://www.anaconda.com/).
If you have any questions about installing Anaconda, you can refer to the [official document](https://docs.anaconda.com/free/anaconda/install/index.html) for more details.

After installing Anaconda, you can create a conda environment for SSSegmentation and activate it, *e.g.*,

```sh
conda create --name ssseg python=3.10 -y
conda activate ssseg
```

For more advanced usages of Anaconda, please also refer to the [official document](https://docs.anaconda.com/free/anaconda/install/index.html).

#### Install Requirements

Now, we can install the necessary requirements in the created environment `ssseg`.

**Step 1: Install PyTorch and Torchvision (Necessary)**

First, it is imperative to install PyTorch and torchvision, which is "Tensors and Dynamic neural networks in Python with strong GPU acceleration". 
Specifically, we recommend the users to follow the [official instructions](https://pytorch.org/get-started/previous-versions/) to install them.

Here, we also provide some example commands about installing PyTorch and Torchvision,

```sh
# conda
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
# CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch
# OSX
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
# ROCM 5.2 (Linux only)
pip install torch==1.13.0+rocm5.2 torchvision==0.14.0+rocm5.2 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

Please note that, SSSegmentation requires `torch.cuda.is_available()` to be `True` and thus, does not support the cpu-only version' pytorch and torchvision now.

**Step 2: Install Other Basic Requirements (Necessary)**

Next, we should install some other essential third-party packages, 

```sh
# clone the source codes from official repository
git clone https://github.com/CharlesPikachu/sssegmentation
cd sssegmentation
# optional, setting SSSEG_WITH_OPS to 1 will compile the necessary extension modules for some algorithms (such as SAMV2) during installation
export SSSEG_WITH_OPS=1
# install some essential requirements
pip install -r requirements.txt
# install ssseg with develop mode
python setup.py develop
```

With the above commands, these python packages will be installed,

- `cython`: set in requirements/basic.txt,
- `fvcore`: set in requirements/basic.txt,
- `black`: set in requirements/basic.txt,
- `usort`: set in requirements/basic.txt,
- `ufmt`: set in requirements/basic.txt,
- `cityscapesscripts`: set in requirements/basic.txt,
- `pycocotools`: set in requirements/basic.txt,
- `pillow`: set in requirements/basic.txt,
- `pandas`: set in requirements/basic.txt,
- `opencv-python`: set in requirements/basic.txt,
- `dill`: set in requirements/basic.txt,
- `iopath`: set in requirements/basic.txt,
- `matplotlib`: set in requirements/basic.txt,
- `tqdm`: set in requirements/basic.txt,
- `argparse`: set in requirements/basic.txt,
- `numpy`: set in requirements/basic.txt,
- `scipy`: set in requirements/basic.txt,
- `chainercv`: set in requirements/optional.txt.

Since the performance of all models in ssseg is evaluated and reported using `chainercv`, we require the installation of this library in the requirements by default.
However, since this library has not been updated for a long time, some of the latest versions of the dependencies it requires, particularly `numpy`, are not compatible with `chainercv`.
Therefore, you might encounter installation failures at this step. In such cases, there are two possible solutions:

- The first solution is to downgrade `numpy` in your environment to version 1.x, *e.g.*, `pip install numpy==1.26.4`,
- The second solution is to manually remove `chainercv` from the requirements.txt.

It is important to note that using the second solution will involve using our custom-defined `Evaluation` class for model performance testing. 
The results may have slight differences compared to those from `chainercv`, but these differences are generally negligible.

**Step 3: Install MMCV (Optional)**

Since some of algorithms integrated in SSSegmentation rely on MMCV that is a foundational library for computer vision research, you are required to install MMCV if you want to use the following algorithms,

- [Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation - CVPR 2022](https://arxiv.org/pdf/2112.01527.pdf),
- [Focal Loss for Dense Object Detection - ICCV 2017](https://arxiv.org/pdf/1708.02002.pdf),
- [PointRend: Image Segmentation as Rendering - CVPR 2020](https://arxiv.org/pdf/1912.08193.pdf),
- [PSANet: Point-wise Spatial Attention Network for Scene Parsing - ECCV 2018](https://hszhao.github.io/papers/eccv18_psanet.pdf),
- [CCNet: Criss-Cross Attention for Semantic Segmentation - ICCV 2019](https://arxiv.org/pdf/1811.11721.pdf).

Specifically, the users can follow the [official instructions](https://mmcv.readthedocs.io/en/latest/) to install MMCV.

Here, we recommend the users to install the pre-build mmcv-full package according to your CUDA and PyTorch version,

```sh
# mmcv < 2.0.0, mmcv versions include mmcv-full and mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# mmcv > 2.0.0, mmcv versions include mmcv and mmcv-lite
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please note that, if the users do not plan to use these mmcv-dependent algorithms, it is not necessary for you to install mmcv. 

**Step 4: Install Apex (Optional)**

Apex holds NVIDIA-maintained utilities to streamline mixed precision and distributed training in Pytorch.
So, the users can install it to utilize [Mixed Precision (FP16) Training](https://arxiv.org/pdf/1710.03740.pdf) supported by Apex to train the segmentors.

In details, the users can follow the [official instructions](https://nvidia.github.io/apex/) to install Apex.

Also, you can leverage the following commands to install Apex,

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

Please note that, SSSegmentation supports two types of mixed precision training, *i.e.*, `apex` and `pytorch`,

```python
import torch

# use Mixed Precision (FP16) Training supported by Apex
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'apex', 'initialize': {'opt_level': 'O1'}, 'scale_loss': {}}
# use Mixed Precision (FP16) Training supported by Pytorch
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'pytorch', 'autocast': {'dtype': torch.float16}, 'grad_scaler': {}}
```

So, the users can choose to utilize [Mixed Precision (FP16) Training](https://arxiv.org/pdf/1710.03740.pdf) supported by Pytorch to train the segmentors if you find it is difficult to install Apex in your environment.

**Step 5: Install TIMM (Optional)**

Timm is a library containing SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations, and training/evaluation scripts. 
It comes packaged with >700 pretrained models, and is designed to be flexible and easy to use.

SSSegmentation provides support for importing backbone networks from [timm](https://github.com/huggingface/pytorch-image-models) to train the segmentors. 
So, if the users want to leverage this feature, you can follow the [official instructions](https://huggingface.co/docs/timm/index) to install timm.

Of course, you can also simply install it with the following command,

```sh
pip install timm
```

For more details, you can refer to [TIMM official repository](https://github.com/huggingface/pytorch-image-models) and [TIMM official document](https://huggingface.co/docs/timm/index).

**Step 6: Install Albumentations (Optional)**

Albumentations is a Python library for fast and flexible image augmentations.

SSSegmentation provides support for importing data augmentation transforms from [albumentations](https://github.com/albumentations-team/albumentations) to train the segmentors.
Thus, if the users want to utilize this feature, you can follow the [official instructions](https://albumentations.ai/docs/) to install albumentations.

Of course, you can also simply install it with the following command,

```sh
pip install -U albumentations
```

For more details, you can refer to [Albumentations official repository](https://github.com/albumentations-team/albumentations) and [Albumentations official document](https://albumentations.ai/docs/).


## Install SSSegmentation as Third-party Package

If the users just want to use SSSegmentation as a dependency or third-party package, you can install SSSegmentation with pip as following,

```sh
# optional, setting SSSEG_WITH_OPS to 1 will compile the necessary extension modules for some algorithms (such as SAMV2) during installation
export SSSEG_WITH_OPS=1
# from pypi
pip install SSSegmentation
# from Github repository
pip install git+https://github.com/CharlesPikachu/sssegmentation.git
```

Here, we assume that you have installed a suitable version of Python, PyTorch and other optional requirements (*e.g.*, mmcv and timm) in your environment before importing SSSegmentation.


## Prepare Datasets

Except for installing SSSegmentation, you are also required to download the benchmark datasets before training the integrated segmentation frameworks.

#### Supported Dataset List

Here is a summary of the supported benchmark datasets and the corresponding download sources,

| Dataset                | Official Websites                                                                          | Download with Provided Scripts                                                                                                  | Download from Baidu Disk                                                                                                        |
| :-:                    | :-:                                                                                        | :-:                                                                                                                             | :-:                                                                                                                             |
| VSPW                   | [Click](https://www.vspwdataset.com/)                                                      | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh vspw` </details>                                              | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| Supervisely            | [Click](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)   | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh supervisely` </details>                                       | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| Dark Zurich            | [Click](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip) | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh darkzurich` </details>                                        | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| Nighttime Driving      | [Click](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip)      | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh nighttimedriving` </details>                                  | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| CIHP                   | [Click](http://sysu-hcp.net/lip/overview.php)                                              | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh cihp` </details>                                              | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| COCOStuff10k           | [Click](https://github.com/nightrome/cocostuff10k)                                         | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh cocostuff10k` </details>                                      | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| COCOStuff164k          | [Click](https://github.com/nightrome/cocostuff)                                            | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh coco` </details>                                              | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| MHPv1&v2               | [Click](https://lv-mhp.github.io/dataset)                                                  | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh mhpv1` & `bash scripts/prepare_datasets.sh mhpv2` </details>  | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| LIP                    | [Click](http://sysu-hcp.net/lip/)                                                          | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh lip` </details>                                               | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| ADE20k                 | [Click](https://groups.csail.mit.edu/vision/datasets/ADE20K/)                              | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh ade20k` </details>                                            | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| SBUShadow              | [Click](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)        | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh sbushadow` </details>                                         | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| CityScapes             | [Click](https://www.cityscapes-dataset.com/)                                               | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh cityscapes` </details>                                        | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| ATR                    | [Click](http://sysu-hcp.net/lip/overview.php)                                              | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh atr` </details>                                               | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| Pascal Context         | [Click](https://cs.stanford.edu/~roozbeh/pascal-context/)                                  | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh pascalcontext` </details>                                     | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| MS COCO                | [Click](https://cocodataset.org/#home)                                                     | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh coco` </details>                                              | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| HRF                    | [Click](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh hrf` </details>                                               | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| CHASE DB1              | [Click](https://staffnet.kingston.ac.uk/~ku15565/)                                         | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh chase_db1` </details>                                         | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| PASCAL VOC             | [Click](http://host.robots.ox.ac.uk/pascal/VOC/)                                           | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh pascalvoc` </details>                                         | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| DRIVE                  | [Click](https://drive.grand-challenge.org/)                                                | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh drive` </details>                                             | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |
| STARE                  | [Click](http://cecas.clemson.edu/~ahoover/stare/)                                          | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh stare` </details>                                             | <details><summary>URL</summary> `https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw` with access code `fn1i` </details>            |

For easier io reading, we will generate `train.txt/val.txt/test.txt` to record the imageids of training, validation and test images for each dataset.
So, it is recommended to adopt the provided script (*i.e.*, `scripts/prepare_datasets.sh`) to download the supported datasets or download the supported datasets from the provided network disk link rather than official website.

#### Prepare Datasets with Provided Scripts

We strongly recommend users to use the scripts we provide to prepare datasets.
And before using the script, it is necessary for you to install wget (for downloading datasets), 7z (for processing compressed packages) and tar (for processing compressed packages) in the environment.

For example, for Linux users, you can run the following commands to install them,

```sh
# wget
apt-get install wget
# 7z
apt-get install p7zip
# tar
apt-get install tar
```

Note that in fact, most Linux systems will install the above three packages by default.
So you don’t need to run the above command to install them again.

For windows users, you can download the corresponding software installation package to install them,

- 7z: [7z official website](https://sparanoid.com/lab/7z/download.html),
- RAR: [rar official website](https://www.win-rar.com/start.html?&L=0),
- WGET: [wget official website](http://downloads.sourceforge.net/gnuwin32/wget-1.11.4-1-setup.exe?spm=a2c6h.12873639.article-detail.7.3f825677H6sKF2&file=wget-1.11.4-1-setup.exe).

Besides, the windows users also need to install [Cmder](https://cmder.app/) to execute the provided script.

After installing these prerequisites, you can use `scripts/prepare_datasets.sh` to prepare the supported benchmark datasets as following,

```sh
------------------------------------------------------------------------------------
scripts/prepare_datasets.sh - prepare datasets for training and inference of SSSegmentation.
------------------------------------------------------------------------------------
Usage:
    bash scripts/prepare_datasets.sh <dataset name>
Options:
    <dataset name>: The dataset name you want to download and prepare.
                    The keyword should be in ['ade20k', 'lip', 'pascalcontext', 'cocostuff10k',
                                              'pascalvoc', 'cityscapes', 'atr', 'chase_db1',
                                              'cihp', 'hrf', 'drive', 'stare', 'nighttimedriving',
                                              'darkzurich', 'sbushadow', 'supervisely', 'vspw',
                                              'mhpv1', 'mhpv2', 'coco',]
    <-h> or <--help>: Show this message.
Examples:
    If you want to fetch ADE20k dataset, you can run 'bash scripts/prepare_datasets.sh ade20k'.
    If you want to fetch Cityscapes dataset, you can run 'bash scripts/prepare_datasets.sh cityscapes'.
------------------------------------------------------------------------------------
```

For example, if you want to train the segmentors with ADE20K dataset, you can prepare the datasets with the following commands,

```sh
bash scripts/prepare_datasets.sh ade20k
```

If the terminal finally outputs "Download ade20k done.", it means you have downloaded the dataset successfully. 
Otherwise, you may have to check and fix your environment issues before re-executing the provided script.