## Introduction

<a href="https://github.com/openseg-group/OCNet.pytorch">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/ocrnet/ocrnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1909.11065.pdf">OCRNet (ECCV'2020)</a></summary>

```latex
@article{yuan2019object,
    title={Object-contextual representations for semantic segmentation},
    author={Yuan, Yuhui and Chen, Xilin and Wang, Jingdong},
    journal={arXiv preprint arXiv:1909.11065},
    year={2019}
}
```

</details>


## Results

#### PASCAL VOC

| Backbone           | Pretrain               | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                        |
| :-:                | :-:                    | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                             |
| R-50-D8            | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.75% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet50os8_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet50os8_voc.log)    |
| R-101-D8           | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.82% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet101os8_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet101os8_voc.log) |
| HRNetV2p-W18-Small | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 72.80% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w18s_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18s_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18s_voc.log)    |
| HRNetV2p-W18       | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 75.80% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w18_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18_voc.log)       |
| HRNetV2p-W48       | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.60% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w48_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w48_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w48_voc.log)       |

#### ADE20k

| Backbone           | Pretrain               | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:                | :-:                    | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                      |
| R-50-D8            | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.47% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet50os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet50os8_ade20k.log)    |
| R-101-D8           | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 43.99% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet101os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet101os8_ade20k.log) |
| HRNetV2p-W18-Small | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 38.04% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w18s_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18s_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18s_ade20k.log)    |
| HRNetV2p-W18       | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 39.85% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w18_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18_ade20k.log)       |
| HRNetV2p-W48       | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.03% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w48_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w48_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w48_ade20k.log)       |

#### CityScapes

| Backbone           | Pretrain               | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                             |
| :-:                | :-:                    | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                  |
| R-50-D8            | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 79.40% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet50os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet50os8_cityscapes.log)    |
| R-101-D8           | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 80.61% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet101os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_resnet101os8_cityscapes.log) |
| HRNetV2p-W18-Small | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 79.30% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w18s_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18s_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18s_cityscapes.log)    |
| HRNetV2p-W18       | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 80.58% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w18_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w18_cityscapes.log)       |
| HRNetV2p-W48       | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | 81.44% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/ocrnet/ocrnet_hrnetv2w48_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w48_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ocrnet/ocrnet_hrnetv2w48_cityscapes.log)       |


## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**