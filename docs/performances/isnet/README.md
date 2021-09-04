# Introduction
```
@article{jin2021isnet,
  title={ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation},
  author={Jin, Zhenchao and Liu, Bin and Chu, Qi and Yu, Nenghai},
  journal={arXiv preprint arXiv:2108.12382},
  year={2021}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## COCOStuff-10k
| Backbone  | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                 |
| :-:       | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 38.06%/40.16%        | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 40.53%/41.60%        | [model]() &#124; [log]() |
| S-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/32/150 | train/test      | 41.55%/42.08%        | [model]() &#124; [log]() |

## ADE20k
| Backbone  | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                 |
| :-:       | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 44.22%/45.04%        | [model]() &#124; [log]() |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 45.92%/47.31%        | [model]() &#124; [log]() |
| S-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.004/poly/16/180 | train/val       | 46.65%/47.55%        | [model]() &#124; [log]() |

## CityScapes
| Backbone  | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                 |
| :-:       | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | train/val       | 79.32%/80.88%        | [model]() &#124; [log]() |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | train/val       | 80.56%/81.98%        | [model]() &#124; [log]() |
| S-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | train/val       | 78.78%/81.30%        | [model]() &#124; [log]() |

## LIP
| Backbone  | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                 |
| :-:       | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150  | train/val       | 53.14%/53.41%        | [model]() &#124; [log]() |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150  | train/val       | 54.96%/55.41%        | [model]() &#124; [log]() |
| S-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.007/poly/40/150 | train/val       | 56.52%/56.81%        | [model]() &#124; [log]() |