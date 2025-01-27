## Introduction

<a href="https://github.com/baidu-research/DeepBench">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/train.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1710.03740.pdf">Mixed Precision (FP16) Training (ArXiv'2017)</a></summary>

```latex
@article{micikevicius2017mixed,
    title={Mixed precision training},
    author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
    journal={arXiv preprint arXiv:1710.03740},
    year={2017}
}
```

</details>


## Results

#### ADE20k

| Segmentor     | Pretrain               | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :-:           | :-:                    | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                             |
| FCN           | ImageNet-1k-224x224    | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 36.67% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fp16/fcn_resnet50os8_apexfp16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/fcn_resnet50os8_apexfp16_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/fcn_resnet50os8_apexfp16_ade20k.log)                               |
| PSPNet        | ImageNet-1k-224x224    | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.06% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fp16/pspnet_resnet50os8_apexfp16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/pspnet_resnet50os8_apexfp16_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/pspnet_resnet50os8_apexfp16_ade20k.log)                      |
| DeepLabV3     | ImageNet-1k-224x224    | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.54% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fp16/deeplabv3_resnet50os8_apexfp16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3_resnet50os8_apexfp16_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3_resnet50os8_apexfp16_ade20k.log)             |
| DeepLabV3plus | ImageNet-1k-224x224    | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.87% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fp16/deeplabv3plus_resnet50os8_apexfp16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3plus_resnet50os8_apexfp16_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3plus_resnet50os8_apexfp16_ade20k.log) |


## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**

SSSegmentation supports two types of mixed precision training, *i.e.*, `apex` and `pytorch`.

(1) To use mixed precision training supported by [APEX](https://github.com/NVIDIA/apex), you should install apex as follow,

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

(2) To use mixed precision training supported by [Pytorch](https://pytorch.org/), you should install torch with `torch.__version__ >= 1.5.0`, *e.g.*,

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