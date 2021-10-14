# Introduction

<a href="https://github.com/yinmh17/DNL-Semantic-Segmentation">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/dnlnet">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2006.06668.pdf">DNLNet (ECCV'2020)</a></summary>

```latex
@misc{yin2020disentangled,
    title={Disentangled Non-Local Neural Networks},
    author={Minghao Yin and Zhuliang Yao and Yue Cao and Xiu Li and Zheng Zhang and Stephen Lin and Han Hu},
    year={2020},
    booktitle={ECCV}
}
```

</details>


# Results

## PASCAL VOC
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.73% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os8_voc_train.log) |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.36% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os16_voc_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.37% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os8_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os8_voc_train.log) |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.25% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os16_voc_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os16_voc_train.log) |

## ADE20k
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.50% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os8_ade20k_train.log) |
| R-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 40.68% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os16_ade20k_train.log) |
| R-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.88% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os8_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os8_ade20k_train.log) |
| R-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 41.80% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os16_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os16_ade20k_train.log) |

## CityScapes
| Backbone  | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:       | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| R-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.75% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os8_cityscapes_train.log) |
| R-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 77.80% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet50os16_cityscapes_train.log) |
| R-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.64% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os8_cityscapes_train.log) |
| R-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.67% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os16_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_dnlnet/dnlnet_resnet101os16_cityscapes_train.log) |


# More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**