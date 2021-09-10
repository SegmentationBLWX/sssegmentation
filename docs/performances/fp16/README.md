# Introduction
```
@article{micikevicius2017mixed,
    title={Mixed precision training},
    author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
    journal={arXiv preprint arXiv:1710.03740},
    year={2017}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## ADE20k
| Model         | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 36.67% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/fcn_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/fcn_r50_ade20k.log) |
| PSPNet        | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.06% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/pspnet_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/pspnet_r50_ade20k.log) |
| DeepLabV3     | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.54% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3_r50_ade20k.log) |
| DeepLabV3plus | R-50-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.87% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3plus_r50_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fp16/deeplabv3plus_r50_ade20k.log) |


# Notes
To use mixed precision training in sssegmentation, you should install apex as follow:
```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```