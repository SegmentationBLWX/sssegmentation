# Introduction
```
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
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 70.88% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_voc_train.log) |
| R-101     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 72.51% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_voc_train.log) |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 38.16% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_ade20k_train.log) |
| R-101     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 39.85% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_ade20k_train.log) |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.09% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet50_cityscapes_train.log) |
| R-101     | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.39% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_semanticfpn/semanticfpn_resnet101_cityscapes_train.log) |