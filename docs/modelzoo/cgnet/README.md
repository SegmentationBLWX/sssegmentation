## Introduction

<a href="https://github.com/wutianyiRosun/CGNet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/cgnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1811.08201.pdf">CGNet (TIP'2020)</a></summary>

```latex
@article{wu2020cgnet,
    title={Cgnet: A light-weight context guided network for semantic segmentation},
    author={Wu, Tianyi and Tang, Sheng and Zhang, Rui and Cao, Juan and Zhang, Yongdong},
    journal={IEEE Transactions on Image Processing},
    volume={30},
    pages={1169--1179},
    year={2020},
    publisher={IEEE}
}
```

</details>


## Results

#### CityScapes
| Segmentor     | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                     |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                          |
| FCN           | M3N21        | 512x1024   | LR/POLICY/BS/EPOCH: 0.001/poly/16/340 | train/val       | 68.53% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_cgnetm3n21_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_cgnet/fcn_cgnetm3n21_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_cgnet/fcn_cgnetm3n21_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**