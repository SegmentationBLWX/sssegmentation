## Introduction

<a href="https://github.com/hszhao/PSPNet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/pspnet/pspnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1612.01105.pdf">PSPNet (CVPR'2017)</a></summary>

```latex
@inproceedings{zhao2017pyramid,
    title={Pyramid scene parsing network},
    author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={2881--2890},
    year={2017}
}
```

</details>


## Results

#### PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.93% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os8_voc_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.29% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os16_voc_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 79.04% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os8_voc_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.92% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet101os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os16_voc_train.log) |

#### ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                     |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.64% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os8_ade20k_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.23% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os16_ade20k_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.55% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os8_ade20k_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.40% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet101os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os16_ade20k_train.log) |

#### CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                            |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                 |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.05% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os8_cityscapes_train.log)       |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.13% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet50os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet50os16_cityscapes_train.log)    |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.94% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os8_cityscapes_train.log)    |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.43% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_resnet101os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pspnet/pspnet_resnet101os16_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**