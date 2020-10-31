# FCN
```
The implemented details see https://arxiv.org/pdf/1411.4038.pdf.
```


# Results

## PASCAL VOC
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 64.73% | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 66.80% | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 71.10% | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 71.17% | -      |

## ADE20k
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 35.72% | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 35.15% | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.07% | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 38.67% | -      |

## CityScapes
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |