## Introduction

<a href="https://github.com/ycszen/TorchSeg/tree/master/model/bisenet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bisenetv2.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2004.02147.pdf">BiSeNetV2 (IJCV'2021)</a></summary>

```latex
@article{yu2021bisenet,
    title={Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation},
    author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
    journal={International Journal of Computer Vision},
    pages={1--18},
    year={2021},
    publisher={Springer}
}
```

</details>


## Results

#### CityScapes
| Segmentor     | Backbone        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                      |
| :-:           | :-:             | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                           |
| FCN           | BiSeNetV2       | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 74.62% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_bisenetv2_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv2/fcn_bisenetv2_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv2/fcn_bisenetv2_cityscapes_train.log)             |
| FCN           | BiSeNetV2-FP16  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 74.17% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_bisenetv2fp16_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv2/fcn_bisenetv2fp16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_bisenetv2/fcn_bisenetv2fp16_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**