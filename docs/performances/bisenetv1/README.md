# Introduction

<a href="https://github.com/ycszen/TorchSeg/tree/master/model/bisenet">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/backbones">Code Snippet</a>

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


# Results

## CityScapes
| Model         | Backbone             | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:                  | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | BiSeNetV1, R-18-D32  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 75.76% | [model]() &#124; [log]() |
| FCN           | BiSeNetV1, R-50-D32  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 77.58% | [model]() &#124; [log]() |


# More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**