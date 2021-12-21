# Get Started


## Prerequisites
- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+


## Installation
(1) Clone the sssegmentation repository.
```sh 
git clone https://github.com/SegmentationBLWX/sssegmentation.git
cd sssegmentation
```

(2) Install requirements.
```sh
pip install -r requirements.txt
```
For mmcv-full, we recommend you to install the pre-build package as below:
```sh
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
For pytorch and torchvision, we recommend you to follow the [official instructions](https://pytorch.org/get-started/previous-versions/), e.g.,
```sh
# CUDA 11.0
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.2
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```

(3) Build some apis, e.g., coco api (if you don't use coco dataset, this operation is unnecessary).
```sh
cd ssseg/libs
sh make.sh
```

(4) Install sssegmentation (this operation is also optional).
```sh
pip install -e .
```