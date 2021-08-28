# Introduction
```
@inproceedings{wang2018non,
    title={Non-local neural networks},
    author={Wang, Xiaolong and Girshick, Ross and Gupta, Abhinav and He, Kaiming},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={7794--7803},
    year={2018}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.08% | [model]() &#124; [log]() |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.17% | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.89% | [model]() &#124; [log]() |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.48% | [model]() &#124; [log]() |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.15% | [model]() &#124; [log]() |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 41.17% | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.49% | [model]() &#124; [log]() |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.45% | [model]() &#124; [log]() |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.34% | [model]() &#124; [log]() |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.18% | [model]() &#124; [log]() |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.42% | [model]() &#124; [log]() |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.48% | [model]() &#124; [log]() |