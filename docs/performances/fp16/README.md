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
| Segmentor     | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                      |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                           |
| FCN           | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 36.67% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/fcn_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/fcn_r50_ade20k.log)                                         |
| PSPNet        | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.06% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/pspnet_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/pspnet_r50_ade20k.log)                             |
| DeepLabV3     | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.54% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3_r50_ade20k.log)                 |
| DeepLabV3plus | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.87% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3plus/deeplabv3plus_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3plus_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3plus_r50_ade20k.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**

To use mixed precision training in sssegmentation, you should install apex as follow:
```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Then, you need to turn on mixed precision training in corresponding config file as follow:
```python
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'apex', 'opt_level': 'O1'}
```