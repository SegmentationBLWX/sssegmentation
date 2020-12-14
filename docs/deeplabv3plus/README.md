# Deeplabv3Plus
```
@inproceedings{chen2018encoder,
  title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={801--818},
  year={2018}
}
The implemented details see https://arxiv.org/pdf/1802.02611.pdf.
```


# Results

## PASCAL VOC
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.43% | -           |
| R-50-D16 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.92% | -           |
| R-101-D8 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 79.19% | -           |
| R-101-D16 (PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.31% | -           |

## ADE20k
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.51% | -           |
| R-50-D16 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.21% | -           |
| R-101-D8 (PRE=ImageNet)  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 45.72% | -           |
| R-101-D16 (PRE=ImageNet) | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 45.22% | -           |

## CityScapes
| Backbone                 | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download    |
| :-:                      | :-:        | :-:                                  | :-:             | :-:    | :-:         |
| R-50-D8 (PRE=ImageNet)   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.38% | -           |
| R-50-D16 (PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.73% | -           |
| R-101-D8 (PRE=ImageNet)  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 81.09% | -           |
| R-101-D16 (PRE=ImageNet) | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.20% | -           |