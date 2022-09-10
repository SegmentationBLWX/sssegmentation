## Introduction

<a href="https://github.com/speedinghzl/CCNet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/ccnet/ccnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1811.11721.pdf">CCNet (ICCV'2019)</a></summary>

```latex
@article{huang2018ccnet,
    title={CCNet: Criss-Cross Attention for Semantic Segmentation},
    author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
    booktitle={ICCV},
    year={2019}
}
```

</details>


## Results

#### PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.43% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os8_voc_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.01% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet50os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os16_voc_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.02% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os8_voc_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.33% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet101os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os16_voc_train.log) |

#### ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                          |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                               |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.47% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os8_ade20k_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.78% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet50os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os16_ade20k_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.00% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os8_ade20k_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.95% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet101os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os16_ade20k_train.log) |

#### CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                      |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                           |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.15% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os8_cityscapes_train.log)       |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.94% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet50os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet50os16_cityscapes_train.log)    |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.08% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os8_cityscapes_train.log)    |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.45% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ccnet/ccnet_resnet101os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ccnet/ccnet_resnet101os16_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**