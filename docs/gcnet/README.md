# CCNet
```
@inproceedings{cao2019gcnet,
    title={Gcnet: Non-local networks meet squeeze-excitation networks and beyond},
    author={Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
    pages={0--0},
    year={2019}
}
The implemented details see https://arxiv.org/pdf/1904.11492.pdf.
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
| R-101-D8 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.19% | -           |
| R-101-D16 (PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |

## CityScapes
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| R-50-D16 (PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| R-101-D8 (PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| R-101-D16 (PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |