## Introduction

<a href="https://github.com/facebookresearch/detectron2">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/semanticfpn/semanticfpn.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1901.02446.pdf">Semantic FPN (CVPR'2019)</a></summary>

```latex
@article{Kirillov_2019,
    title={Panoptic Feature Pyramid Networks},
    ISBN={9781728132938},
    url={http://dx.doi.org/10.1109/CVPR.2019.00656},
    DOI={10.1109/cvpr.2019.00656},
    journal={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    publisher={IEEE},
    author={Kirillov, Alexander and Girshick, Ross and He, Kaiming and Dollar, Piotr},
    year={2019},
    month={Jun}
}
```

</details>


## Results

#### PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                         |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                              |
| R-50      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 70.88% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/semanticfpn/semanticfpn_resnet50_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_voc_train.log)    |
| R-101     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 72.51% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/semanticfpn/semanticfpn_resnet101_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_voc_train.log) |

#### ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                       |
| R-50      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 38.16% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/semanticfpn/semanticfpn_resnet50_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_ade20k_train.log)    |
| R-101     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 39.85% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/semanticfpn/semanticfpn_resnet101_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_ade20k_train.log) |

#### CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| R-50      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.09% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/semanticfpn/semanticfpn_resnet50_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_cityscapes_train.log)    |
| R-101     | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.39% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/semanticfpn/semanticfpn_resnet101_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**