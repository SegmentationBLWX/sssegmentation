## Introduction

<a href="http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/unet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1505.04597.pdf">UNet (MICCAI'2016/Nat. Methods'2019)</a></summary>

```latex
@inproceedings{ronneberger2015u,
    title={U-net: Convolutional networks for biomedical image segmentation},
    author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
    booktitle={International Conference on Medical image computing and computer-assisted intervention},
    pages={234--241},
    year={2015},
    organization={Springer}
}
```

</details>


## Results

#### HRF
| Segmentor     | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                                                                                                                                                                                                                                                                                                                                                                                      |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                           |
| FCN           | UNet-S5-D16  | 256x256    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 79.88% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_unets5os16_hrf.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_hrf_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_hrf_train.log)                         |
| PSPNet        | UNet-S5-D16  | 256x256    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.26% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_unets5os16_hrf.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_hrf_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_hrf_train.log)             |
| DeepLabV3     | UNet-S5-D16  | 256x256    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.29% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_unets5os16_hrf.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_hrf_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_hrf_train.log) |

#### DRIVE
| Segmentor     | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                                                                                                                                                                                                                                                                                                                                                                                            |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                 |
| FCN           | UNet-S5-D16  | 64x64      | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 78.67% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_unets5os16_drive.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_drive_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_drive_train.log)                         |
| PSPNet        | UNet-S5-D16  | 64x64      | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 78.77% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_unets5os16_drive.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_drive_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_drive_train.log)             |
| DeepLabV3     | UNet-S5-D16  | 64x64      | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 78.96% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_unets5os16_drive.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_drive_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_drive_train.log) |

#### STARE
| Segmentor     | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                                                                                                                                                                                                                                                                                                                                                                                            |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                 |
| FCN           | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 81.03% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_unets5os16_stare.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_stare_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_stare_train.log)                         |
| PSPNet        | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 81.24% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_unets5os16_stare.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_stare_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_stare_train.log)             |
| DeepLabV3     | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 81.19% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_unets5os16_stare.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_stare_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_stare_train.log) |

#### CHASE DB1
| Segmentor     | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                                                                                                                                                                                                                                                                                                                                                                                                     |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                          |
| FCN           | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.50% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_unets5os16_chasedb1.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_chasedb1_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/fcn_unets5os16_chasedb1_train.log)                         |
| PSPNet        | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.50% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_unets5os16_chasedb1.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_chasedb1_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/pspnet_unets5os16_chasedb1_train.log)             |
| DeepLabV3     | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.54% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_unets5os16_chasedb1.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_chasedb1_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_unet/deeplabv3_unets5os16_chasedb1_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**