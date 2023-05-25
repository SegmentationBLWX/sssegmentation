## Introduction

<a href="https://github.com/ycszen/TorchSeg/tree/master/model/bisenet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bisenetv1.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1808.00897.pdf">BiSeNetV1 (ECCV'2018)</a></summary>

```latex
@inproceedings{yu2018bisenet,
    title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
    author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    pages={325--341},
    year={2018}
}
```

</details>


## Results

#### CityScapes
| Segmentor     | Backbone             | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:           | :-:                  | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| FCN           | BiSeNetV1, R-18-D32  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 75.76% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_bisenetv1_resnet18os32_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv1/fcn_bisenetv1_resnet18os32_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv1/fcn_bisenetv1_resnet18os32_cityscapes_train.log) |
| FCN           | BiSeNetV1, R-50-D32  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 77.78% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_bisenetv1_resnet50os32_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv1/fcn_bisenetv1_resnet50os32_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv1/fcn_bisenetv1_resnet50os32_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**