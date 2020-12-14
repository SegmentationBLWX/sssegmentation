# CCNet
```
@article{huang2018ccnet,
    title={CCNet: Criss-Cross Attention for Semantic Segmentation},
    author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
    booktitle={ICCV},
    year={2019}
}
The implemented details see https://arxiv.org/pdf/1811.11721.pdf.
```


# Results

## PASCAL VOC
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| R-50-D16 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| R-101-D8 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| R-101-D16 (PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |

## ADE20k
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |
| R-50-D16 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |
| R-101-D8 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |
| R-101-D16 (PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |

## CityScapes
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| R-50-D16 (PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| R-101-D8 (PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| R-101-D16 (PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |