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


## Installation

**1.Clone SSSegmentation**

You can run the following commands to clone the sssegmentation repository,

```sh 
git clone https://github.com/SegmentationBLWX/sssegmentation.git
cd sssegmentation
```

**2.Install Requirements**

You can run the following commands to install the necessary requirements,

```sh
pip install -r requirements.txt
```

For mmcv-full, we recommend you to install the pre-build package as below,

```sh
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

For pytorch and torchvision, we recommend you to follow the [official instructions](https://pytorch.org/get-started/previous-versions/), *e.g.*,

```sh
# CUDA 11.0
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.2
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```

If you want to install apex for [Mixed Precision (FP16) Training](https://arxiv.org/pdf/1710.03740.pdf), we recommend you to follow the instructions in [official repository](https://github.com/NVIDIA/apex).
For example, the Linux users can install apex with the following commands,

```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```