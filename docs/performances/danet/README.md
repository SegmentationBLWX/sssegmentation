## Introduction

<a href="https://github.com/junfu1115/DANet/">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/danet/danet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1809.02983.pdf">DANet (CVPR'2019)</a></summary>

```latex
@article{fu2018dual,
    title={Dual Attention Network for Scene Segmentation},
    author={Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```

</details>


## Results

#### PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.39% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os8_voc_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.04% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet50os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os16_voc_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.97% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os8_voc_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.99% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet101os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os16_voc_train.log) |

#### ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                          |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                               |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.90% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os8_ade20k_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.85% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet50os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os16_ade20k_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.37% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os8_ade20k_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.58% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet101os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os16_ade20k_train.log) |

#### CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                      |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                           |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.47% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os8_cityscapes_train.log)       |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.60% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet50os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet50os16_cityscapes_train.log)    |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.55% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os8_cityscapes_train.log)    |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.23% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/danet/danet_resnet101os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_danet/danet_resnet101os16_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**