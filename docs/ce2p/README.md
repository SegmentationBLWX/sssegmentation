# CE2P
```
The implemented details see https://arxiv.org/pdf/1809.05996.pdf.
```


# Results

## PASCAL VOC
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |

## LIP
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | -      | -      |