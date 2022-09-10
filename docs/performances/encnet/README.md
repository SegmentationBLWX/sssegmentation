## Introduction

<a href="https://github.com/zhanghang1989/PyTorch-Encoding">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/encnet/encnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1803.08904.pdf">EncNet (CVPR'2018)</a></summary>

```latex
@InProceedings{Zhang_2018_CVPR,
    author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
    title = {Context Encoding for Semantic Segmentation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```

</details>


## Results

#### PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.53% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os8_voc_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 74.55% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet50os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os16_voc_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.61% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os8_voc_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.41% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet101os16_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os16_voc_train.log) |

#### ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                     |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.60% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os8_ade20k_train.log)       |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 39.70% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet50os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os16_ade20k_train.log)    |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.43% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os8_ade20k_train.log)    |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 41.65% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet101os16_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os16_ade20k_train.log) |

#### CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                            |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                 |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.98% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os8_cityscapes_train.log)       |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 75.98% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet50os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet50os16_cityscapes_train.log)    |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.70% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os8_cityscapes_train.log)    |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.46% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/encnet/encnet_resnet101os16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_encnet/encnet_resnet101os16_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**