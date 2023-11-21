## Introduction

<a href="https://github.com/openseg-group/openseg.pytorch">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/isanet/isanet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1907.12273.pdf">ISANet (ArXiv'2019/IJCV'2021)</a></summary>

```latex
@article{huang2019isa,
    title={Interlaced Sparse Self-Attention for Semantic Segmentation},
    author={Huang, Lang and Yuan, Yuhui and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
    journal={arXiv preprint arXiv:1907.12273},
    year={2019}
}
```

</details>


## Results

#### PASCAL VOC

| Backbone  | Pretrain               | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                        |
| :-:       | :-:                    | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                             |
| R-50-D8   | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 76.99% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isanet/isanet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_voc.log)    |
| R-101-D8  | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 78.60% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isanet/isanet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_voc.log) |

#### ADE20k

| Backbone  | Pretrain               | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:       | :-:                    | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                      |
| R-50-D8   | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 42.60% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isanet/isanet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_ade20k.log)    |
| R-101-D8  | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 44.08% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isanet/isanet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_ade20k.log) |

#### CityScapes

| Backbone  | Pretrain               | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                             |
| :-:       | :-:                    | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                  |
| R-50-D8   | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.34% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isanet/isanet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet50os8_cityscapes.log)    |
| R-101-D8  | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.58% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isanet/isanet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isanet/isanet_resnet101os8_cityscapes.log) |


## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**