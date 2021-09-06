# Introduction
```
@inproceedings{chen2018encoder,
    title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
    author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    pages={801--818},
    year={2018}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.43% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os8_voc_train.log) |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.92% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os16_voc_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 79.19% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os8_voc_train.log) |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.31% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os16_voc_train.log) |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.51% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os8_ade20k_train.log) |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.21% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os16_ade20k_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 45.72% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os8_ade20k_train.log) |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 45.22% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os16_ade20k_train.log) |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.38% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os8_cityscapes_train.log) |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.73% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet50os16_cityscapes_train.log) |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 81.09% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os8_cityscapes_train.log) |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.20% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_deeplabv3plus/deeplabv3plus_resnet101os16_cityscapes_train.log) |