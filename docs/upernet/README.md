# UPerNet
```
@inproceedings{xiao2018unified,
  title={Unified perceptual parsing for scene understanding},
  author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={418--434},
  year={2018}
}
The implemented details see https://arxiv.org/pdf/1807.10221.pdf.
```


# Results

## PASCAL VOC
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -           |

## ADE20k
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.02% | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | -           |

## CityScapes
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |
| ResNet101 (OS=16, PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -           |