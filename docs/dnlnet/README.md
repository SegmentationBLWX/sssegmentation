# CCNet
```
@misc{yin2020disentangled,
    title={Disentangled Non-Local Neural Networks},
    author={Minghao Yin and Zhuliang Yao and Yue Cao and Xiu Li and Zheng Zhang and Stephen Lin and Han Hu},
    year={2020},
    booktitle={ECCV}
}
The implemented details see https://arxiv.org/pdf/2006.06668.pdf.
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