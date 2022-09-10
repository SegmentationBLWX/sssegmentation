## Introduction

<a href="https://github.com/wuhuikai/FastFCN">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/fastfcn/fastfcn.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1903.11816.pdf">FastFCN (ArXiv'2019)</a></summary>

```latex
@article{wu2019fastfcn,
    title={Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation},
    author={Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu, Yizhou},
    journal={arXiv preprint arXiv:1903.11816},
    year={2019}
}
```

</details>


## Results

#### CityScapes
| Segmentor     | Backbone  | Crop Size  | Schedule                            | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| :-:           | :-:       | :-:        | :-:                                 | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| EncNet        | R-50-D32  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220 | train/val       | 78.42% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fastfcn/fastfcn_encnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_encnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_encnet_resnet50os8_cityscapes_train.log)          |
| PSPNet        | R-50-D32  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220 | train/val       | 79.36% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fastfcn/fastfcn_pspnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_pspnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_pspnet_resnet50os8_cityscapes_train.log)          |
| DeepLabV3     | R-50-D32  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220 | train/val       | 79.96% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fastfcn/fastfcn_deeplabv3_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_deeplabv3_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_deeplabv3_resnet50os8_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**