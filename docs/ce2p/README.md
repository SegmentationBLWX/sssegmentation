# Introduction
```
@inproceedings{ruan2019devil,
  title={Devil in the details: Towards accurate single and multiple human parsing},
  author={Ruan, Tao and Liu, Ting and Huang, Zilong and Wei, Yunchao and Wei, Shikui and Zhao, Yao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={4814--4821},
  year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1nPxHw5Px7a7jZMiX-ZxRhA (code is 3jvr)
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.69% | [model]() &#124; [log]() |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 74.58% | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.77% | [model]() &#124; [log]() |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.84% | [model]() &#124; [log]() |

## LIP
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 52.42% | [model]() &#124; [log]() |
| R-50-D16  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 51.98% | [model]() &#124; [log]() |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.79% | [model]() &#124; [log]() |
| R-101-D16 | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.02% | [model]() &#124; [log]() |

## CIHP
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 61.15% | [model]() &#124; [log]() |
| R-50-D16  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 60.15% | [model]() &#124; [log]() |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 63.83% | [model]() &#124; [log]() |
| R-101-D16 | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 62.25% | [model]() &#124; [log]() |

## ATR
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.02% | [model]() &#124; [log]() |
| R-50-D16  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 77.62% | [model]() &#124; [log]() |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.57% | [model]() &#124; [log]() |
| R-101-D16 | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.25% | [model]() &#124; [log]() |