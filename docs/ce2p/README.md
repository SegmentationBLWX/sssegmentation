# CE2P
```
@inproceedings{ruan2019devil,
  title={Devil in the details: Towards accurate single and multiple human parsing},
  author={Ruan, Tao and Liu, Ting and Huang, Zilong and Wei, Yunchao and Wei, Shikui and Zhao, Yao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={4814--4821},
  year={2019}
}
The implemented details see https://arxiv.org/pdf/1809.05996.pdf.
```


# Results

## PASCAL VOC
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.69% | -           |
| R-50-D16 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 74.58% | -           |
| R-101-D8 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.77% | -           |
| R-101-D16 (PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.84% | -           |

## LIP
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 52.42% | -           |
| R-50-D16 (PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 51.98% | -           |
| R-101-D8 (PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.79% | -           |
| R-101-D16 (PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.02% | -           |

## CIHP
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 61.15% | -           |
| R-50-D16 (PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 60.15% | -           |
| R-101-D8 (PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 63.83% | -           |
| R-101-D16 (PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 62.25% | -           |

## ATR
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.02% | -           |
| R-50-D16 (PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 77.62% | -           |
| R-101-D8 (PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.57% | -           |
| R-101-D16 (PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.25% | -           |