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
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.06% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 74.44% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.07% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.08% | -           |

## LIP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 52.82% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 51.83% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.83% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 53.93% | -           |

## CIHP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 61.31% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 59.64% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 63.64% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 61.96% | -           |

## ATR
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.05% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 77.57% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.58% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.44% | -           |