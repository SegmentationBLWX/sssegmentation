# Introduction
```
@article{wu2020cgnet,
    title={Cgnet: A light-weight context guided network for semantic segmentation},
    author={Wu, Tianyi and Tang, Sheng and Zhang, Rui and Cao, Juan and Zhang, Yongdong},
    journal={IEEE Transactions on Image Processing},
    volume={30},
    pages={1169--1179},
    year={2020},
    publisher={IEEE}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## CityScapes
| Model         | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:    | :-:                      |
| FCN           | M3N21        | 512x1024   | LR/POLICY/BS/EPOCH: 0.001/poly/16/340 | train/val       | 68.53% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_cgnet/fcn_cgnetm3n21_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_cgnet/fcn_cgnetm3n21_cityscapes_train.log) |