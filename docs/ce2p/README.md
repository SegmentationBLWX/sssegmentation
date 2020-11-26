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
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |

## LIP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.58% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.18% | -           |

## CIHP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |

## ATR
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |