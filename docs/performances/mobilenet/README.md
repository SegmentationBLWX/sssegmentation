## Introduction

<a href="https://github.com/tensorflow/models/tree/master/research/deeplab">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/mobilenet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1801.04381.pdf">MobileNetV2 (CVPR'2018)</a></summary>

```latex
@inproceedings{sandler2018mobilenetv2,
    title={Mobilenetv2: Inverted residuals and linear bottlenecks},
    author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={4510--4520},
    year={2018}
}
```

</details>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1905.02244.pdf">MobileNetV3 (ICCV'2019)</a></summary>

```latex
@inproceedings{Howard_2019_ICCV,
    title={Searching for MobileNetV3},
    author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    pages={1314-1324},
    month={October},
    year={2019},
    doi={10.1109/ICCV.2019.00140}}
}
```

</details>


## Results

#### PASCAL VOC
| Segmentor     | Backbone | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                            |
| :-:           | :-:      | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| FCN           | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 59.89% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_mobilenetv2os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/fcn_mobilenetv2os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/fcn_mobilenetv2os8_voc_train.log)                                         |
| PSPNet        | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 68.40% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_mobilenetv2os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/pspnet_mobilenetv2os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/pspnet_mobilenetv2os8_voc_train.log)                             |
| DeepLabV3     | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 70.08% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_mobilenetv2os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3_mobilenetv2os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3_mobilenetv2os8_voc_train.log)                 |
| DeepLabV3plus | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 70.04% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3plus/deeplabv3plus_mobilenetv2os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3plus_mobilenetv2os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3plus_mobilenetv2os8_voc_train.log) |
| LRASPPNet     | M-V3S-D8 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/180 | trainaug/val    | 62.13% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/lrasppnet/lrasppnet_mobilenetv3sos8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3sos8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3sos8_voc_train.log)              |
| LRASPPNet     | M-V3L-D8 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/180 | trainaug/val    | 67.90% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/lrasppnet/lrasppnet_mobilenetv3los8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3los8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3los8_voc_train.log)              |

#### ADE20k
| Segmentor     | Backbone | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :-:           | :-:      | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| FCN           | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 30.85% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_mobilenetv2os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/fcn_mobilenetv2os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/fcn_mobilenetv2os8_ade20k_train.log)                                         |
| PSPNet        | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 35.09% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_mobilenetv2os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/pspnet_mobilenetv2os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/pspnet_mobilenetv2os8_ade20k_train.log)                             |
| DeepLabV3     | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 37.55% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_mobilenetv2os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3_mobilenetv2os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3_mobilenetv2os8_ade20k_train.log)                 |
| DeepLabV3plus | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 37.66% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3plus/deeplabv3plus_mobilenetv2os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3plus_mobilenetv2os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3plus_mobilenetv2os8_ade20k_train.log) |
| LRASPPNet     | M-V3S-D8 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/390 | train/val       | 26.09% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/lrasppnet/lrasppnet_mobilenetv3sos8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3sos8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3sos8_ade20k_train.log)              |
| LRASPPNet     | M-V3L-D8 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/390 | train/val       | 30.06% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/lrasppnet/lrasppnet_mobilenetv3los8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3los8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3los8_ade20k_train.log)              |

#### CityScapes
| Segmentor     | Backbone | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:           | :-:      | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| FCN           | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 70.77% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_mobilenetv2os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/fcn_mobilenetv2os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/fcn_mobilenetv2os8_cityscapes_train.log)                                         |
| PSPNet        | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 73.64% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/pspnet/pspnet_mobilenetv2os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/pspnet_mobilenetv2os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/pspnet_mobilenetv2os8_cityscapes_train.log)                             |
| DeepLabV3     | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.74% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3/deeplabv3_mobilenetv2os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3_mobilenetv2os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3_mobilenetv2os8_cityscapes_train.log)                 |
| DeepLabV3plus | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.68% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/deeplabv3plus/deeplabv3plus_mobilenetv2os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3plus_mobilenetv2os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/deeplabv3plus_mobilenetv2os8_cityscapes_train.log) |
| LRASPPNet     | M-V3S-D8 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/660 | train/val       | 65.06% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/lrasppnet/lrasppnet_mobilenetv3sos8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3sos8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3sos8_cityscapes_train.log)              |
| LRASPPNet     | M-V3L-D8 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/660 | train/val       | 69.98% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/lrasppnet/lrasppnet_mobilenetv3los8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3los8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilenet/lrasppnet_mobilenetv3los8_cityscapes_train.log)              |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**