## Introduction

<a href="https://github.com/hszhao/ICNet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/icnet/icnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1704.08545.pdf">ICNet (ECCV'2018)</a></summary>

```latex
@inproceedings{zhao2018icnet,
    title={Icnet for real-time semantic segmentation on high-resolution images},
    author={Zhao, Hengshuang and Qi, Xiaojuan and Shen, Xiaoyong and Shi, Jianping and Jia, Jiaya},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    pages={405--420},
    year={2018}
}
```

</details>


## Results

#### CityScapes
| Backbone  | Crop Size  | Schedule                            | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                   |
| :-:       | :-:        | :-:                                 | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                        |
| R-50-D8   | 832x832    | LR/POLICY/BS/EPOCH: 0.01/poly/8/440 | train/val       | 76.60% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/icnet/icnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet50os8_cityscapes_train.log)    |
| R-101-D8  | 832x832    | LR/POLICY/BS/EPOCH: 0.01/poly/8/440 | train/val       | 76.27% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/icnet/icnet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_icnet/icnet_resnet101os8_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**