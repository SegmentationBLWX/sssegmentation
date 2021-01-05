# Introduction
```
@article{yuan2019object,
    title={Object-contextual representations for semantic segmentation},
    author={Yuan, Yuhui and Chen, Xilin and Wang, Jingdong},
    journal={arXiv preprint arXiv:1909.11065},
    year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1nPxHw5Px7a7jZMiX-ZxRhA (code is 3jvr)
```


# Results

## PASCAL VOC
| Backbone           | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:                | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8            | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.75% | [model]() &#124; [log]() |
| R-101-D8           | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.82% | [model]() &#124; [log]() |
| HRNetV2p-W18-Small | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 72.80% | [model]() &#124; [log]() |
| HRNetV2p-W18       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.80% | [model]() &#124; [log]() |
| HRNetV2p-W48       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.60% | [model]() &#124; [log]() |

## ADE20k
| Backbone           | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:                | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8            | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.47% | [model]() &#124; [log]() |
| R-101-D8           | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.99% | [model]() &#124; [log]() |
| HRNetV2p-W18-Small | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 38.04% | [model]() &#124; [log]() |
| HRNetV2p-W18       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 39.85% | [model]() &#124; [log]() |
| HRNetV2p-W48       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.03% | [model]() &#124; [log]() |

## CityScapes
| Backbone           | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:                | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8            | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 79.40% | [model]() &#124; [log]() |
| R-101-D8           | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 80.61% | [model]() &#124; [log]() |
| HRNetV2p-W18-Small | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 79.30% | [model]() &#124; [log]() |
| HRNetV2p-W18       | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 80.58% | [model]() &#124; [log]() |
| HRNetV2p-W48       | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 81.44% | [model]() &#124; [log]() |