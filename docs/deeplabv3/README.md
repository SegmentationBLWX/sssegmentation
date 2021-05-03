# Introduction
```
@article{chen2017rethinking,
    title={Rethinking atrous convolution for semantic image segmentation},
    author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
    journal={arXiv preprint arXiv:1706.05587},
    year={2017}
}
All the reported models here are available at https://pan.baidu.com/s/1nPxHw5Px7a7jZMiX-ZxRhA (code is 3jvr)
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.72% | [model]() &#124; [log]() |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.86% | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 79.52% | [model]() &#124; [log]() |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.55% | [model]() &#124; [log]() |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.19% | [model]() &#124; [log]() |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 41.41% | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 45.16% | [model]() &#124; [log]() |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.45% | [model]() &#124; [log]() |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.62% | [model]() &#124; [log]() |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.19% | [model]() &#124; [log]() |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.28% | [model]() &#124; [log]() |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.03% | [model]() &#124; [log]() |

## PASCAL Context
| Backbone  | Crop Size  | Schedule                               | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                    | :-:             | :-:    | :-:                      |
| R-50-D8   | 480x480    | LR/POLICY/BS/EPOCH: 0.004/poly/16/260  | train/val       | 46.31% | [model]() &#124; [log]() |
| R-101-D8  | 480x480    | LR/POLICY/BS/EPOCH: 0.004/poly/16/260  | train/val       | 48.43% | [model]() &#124; [log]() |

## PASCAL Context 59
| Backbone  | Crop Size  | Schedule                               | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                    | :-:             | :-:    | :-:                      |
| R-50-D8   | 480x480    | LR/POLICY/BS/EPOCH: 0.004/poly/16/260  | train/val       | 51.69% | [model]() &#124; [log]() |
| R-101-D8  | 480x480    | LR/POLICY/BS/EPOCH: 0.004/poly/16/260  | train/val       | 53.81% | [model]() &#124; [log]() |