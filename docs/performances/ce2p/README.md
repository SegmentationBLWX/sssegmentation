# Introduction
```
@inproceedings{ruan2019devil,
    title={Devil in the details: Towards accurate single and multiple human parsing},
    author={Ruan, Tao and Liu, Ting and Huang, Zilong and Wei, Yunchao and Wei, Shikui and Zhao, Yao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={33},
    pages={4814--4821},
    year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.69% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_voc_train.log) |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 74.58% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_voc_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.77% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_voc_train.log) |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.84% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_voc_train.log) |

## LIP
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 52.42% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_lip_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_lip_train.log) |
| R-50-D16  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 51.98% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_lip_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_lip_train.log) |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.79% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_lip_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_lip_train.log) |
| R-101-D16 | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 54.02% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_lip_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_lip_train.log) |

## CIHP
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 61.15% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_cihp_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_cihp_train.log) |
| R-50-D16  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 60.15% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_cihp_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_cihp_train.log) |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 63.83% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_cihp_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_cihp_train.log) |
| R-101-D16 | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 62.25% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_cihp_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_cihp_train.log) |

## ATR
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.02% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_atr_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_atr_train.log) |
| R-50-D16  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 77.62% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_atr_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os16_atr_train.log) |
| R-101-D8  | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.57% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_atr_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os8_atr_train.log) |
| R-101-D16 | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150 | train/val       | 78.25% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_atr_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet101os16_atr_train.log) |