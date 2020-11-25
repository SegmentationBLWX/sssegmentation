# PSPNet
```
@inproceedings{zhao2017pyramid,
  title={Pyramid scene parsing network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2881--2890},
  year={2017}
}
The implemented details see https://arxiv.org/pdf/1612.01105.pdf.
```


# Results

## PASCAL VOC
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.93% | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | -      |

## ADE20k
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.10% | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.23% | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.55% | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.40% | -      |

## CityScapes
| Backbone                        | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | FPS    |
| :-:                             | :-:        | :-:                                  | :-:             | :-:    | :-:    |
| ResNet50 (OS=8, PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet50 (OS=16, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet101 (OS=8, PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |
| ResNet101 (OS=16, PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | -      |