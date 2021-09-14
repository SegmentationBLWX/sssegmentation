# Introduction
```
@inproceedings{kirillov2020pointrend,
    title={Pointrend: Image segmentation as rendering},
    author={Kirillov, Alexander and Wu, Yuxin and He, Kaiming and Girshick, Ross},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    pages={9799--9808},
    year={2020}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 69.84% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet50os8_voc_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 72.31% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet101os8_voc_train.log) |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 37.80% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet50os8_ade20k_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.26% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet101os8_ade20k_train.log) |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 76.89% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet50os8_cityscapes_train.log) |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.80% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_pointrend/pointrend_resnet101os8_cityscapes_train.log) |