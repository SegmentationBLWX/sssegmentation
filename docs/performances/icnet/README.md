# Introduction
```
@inproceedings{zhao2018icnet,
    title={Icnet for real-time semantic segmentation on high-resolution images},
    author={Zhao, Hengshuang and Qi, Xiaojuan and Shen, Xiaoyong and Shi, Jianping and Jia, Jiaya},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    pages={405--420},
    year={2018}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## CityScapes
| Backbone  | Crop Size  | Schedule                            | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                 | :-:             | :-:    | :-:                      |
| R-50-D8   | 832x832    | LR/POLICY/BS/EPOCH: 0.01/poly/8/440 | train/val       | 76.60% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet50os8_cityscapes_train.log) |
| R-101-D8  | 832x832    | LR/POLICY/BS/EPOCH: 0.01/poly/8/440 | train/val       | 76.27% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet101os8_cityscapes_train.log) |