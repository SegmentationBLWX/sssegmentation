# OCRNet
```
@article{yuan2019object,
  title={Object-contextual representations for semantic segmentation},
  author={Yuan, Yuhui and Chen, Xilin and Wang, Jingdong},
  journal={arXiv preprint arXiv:1909.11065},
  year={2019}
}
The implemented details see https://arxiv.org/pdf/1909.11065.pdf.
```


# Results

## PASCAL VOC
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.29% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.11% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.72% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.54% | -           |

## LIP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 52.39% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 51.54% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.80% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 53.12% | -           |

## CIHP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 59.65% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 57.00% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 62.66% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 59.74% | -           |

## ATR
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 77.20% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 74.30% | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 77.92% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 75.00% | -           |