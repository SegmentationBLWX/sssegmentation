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
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.72% | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.54% | -           |

## LIP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -           |

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