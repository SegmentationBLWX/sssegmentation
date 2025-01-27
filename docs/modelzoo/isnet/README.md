## Introduction

<a href="https://github.com/SegmentationBLWX/sssegmentation">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/isnet/isnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2108.12382.pdf">ISNet (ICCV'2021)</a></summary>

```latex
@inproceedings{jin2021isnet,
    title={ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation},
    author={Jin, Zhenchao and Liu, Bin and Chu, Qi and Yu, Nenghai},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={7189--7198},
    year={2021}
}
```

</details>


## Results

#### COCOStuff-10k

| Backbone  | Pretrain               | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                                                                                                                                                                                                                                                                                                                                                                                                |
| :-:       | :-:                    | :-:        | :-:                                   | :-:             | :-:                  | :-:                                                                                                                                                                                                                                                                                                                                                                                                     |
| R-50-D8   | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 38.06%/40.39%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet50os8_cocostuff10k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_cocostuff10k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_cocostuff10k.log)       |
| R-101-D8  | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 40.53%/41.74%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet101os8_cocostuff10k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_cocostuff10k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_cocostuff10k.log)    |
| S-101-D8  | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/32/150 | train/test      | 41.55%/42.53%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnest101os8_cocostuff10k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_cocostuff10k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_cocostuff10k.log) |

#### ADE20k

| Backbone  | Pretrain               | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                                                                                                                                                                                                                                                                                                                                                                              |
| :-:       | :-:                    | :-:        | :-:                                   | :-:             | :-:                  | :-:                                                                                                                                                                                                                                                                                                                                                                                   |
| R-50-D8   | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 44.22%/45.03%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet50os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_ade20k.log)       |
| R-101-D8  | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 45.92%/47.29%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_ade20k.log)    |
| S-101-D8  | ImageNet-1k-224x224    | 512x512    | LR/POLICY/BS/EPOCH: 0.004/poly/16/180 | train/val       | 46.65%/47.56%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnest101os8_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_ade20k.log) |

#### CityScapes

| Backbone  | Pretrain               | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(ms+flip)   | Download                                                                                                                                                                                                                                                                                                                                                                                          |
| :-:       | :-:                    | :-:        | :-:                                   | :-:             | :-:                  | :-:                                                                                                                                                                                                                                                                                                                                                                                               |
| R-50-D8   | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | train/val       | 79.32%/81.31%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet50os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_cityscapes.log)       |
| R-101-D8  | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | train/val       | 80.56%/81.96%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_cityscapes.log)    |
| S-101-D8  | ImageNet-1k-224x224    | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | train/val       | 78.78%/81.33%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnest101os8_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_cityscapes.log) |

#### LIP

| Backbone  | Pretrain               | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU(flip)      | Download                                                                                                                                                                                                                                                                                                                                                                     |
| :-:       | :-:                    | :-:        | :-:                                   | :-:             | :-:                  | :-:                                                                                                                                                                                                                                                                                                                                                                          |
| R-50-D8   | ImageNet-1k-224x224    | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150  | train/val       | 53.14%/53.41%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet50os8_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet50os8_lip.log)       |
| R-101-D8  | ImageNet-1k-224x224    | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150  | train/val       | 54.96%/55.41%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnet101os8_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnet101os8_lip.log)    |
| S-101-D8  | ImageNet-1k-224x224    | 473x473    | LR/POLICY/BS/EPOCH: 0.007/poly/40/150 | train/val       | 56.52%/56.81%        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/isnet/isnet_resnest101os8_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_isnet/isnet_resnest101os8_lip.log) |


## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**

Please note that, due to differences in computational precision, the numerical values obtained when testing model performance on different versions of PyTorch or graphics cards may vary slightly. 
This is a normal phenomenon and the performance differences are generally within 0.1%.