# FCN
```
The implemented details see https://arxiv.org/pdf/1411.4038.pdf.
```


# Results

## PASCAL VOC
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |

## ADE20k
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -      |

## CityScapes
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |