# Introduction
```
@article{huang2019isa,
    title={Interlaced Sparse Self-Attention for Semantic Segmentation},
    author={Huang, Lang and Yuan, Yuhui and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
    journal={arXiv preprint arXiv:1907.12273},
    year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.99% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_voc_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.60% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_voc_train.log) |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.60% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_ade20k_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.08% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_ade20k_train.log) |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.34% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_cityscapes_train.log) |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.58% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_cityscapes_train.log) |