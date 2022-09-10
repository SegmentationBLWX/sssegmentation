## Introduction

<a href="https://github.com/facebookresearch/ConvNeXt">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/convnext.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2201.03545.pdf">ConvNeXt (CVPR'2022)</a></summary>

```latex
@article{liu2022convnet,
    title={A ConvNet for the 2020s},
    author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
    journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

</details>


## Results

#### ADE20k
| Segmentor     | Backbone              | Crop Size  | Schedule                               | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                         |
| :-:           | :-:                   | :-:        | :-:                                    | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                              |
| UperNet       | ConvNeXt-T            | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 46.25% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnexttiny_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnexttiny_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnexttiny_ade20k.log)                |
| UperNet       | ConvNeXt-S            | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 48.68% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextsmall_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextsmall_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextsmall_ade20k.log)             |
| UperNet       | ConvNeXt-B            | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 48.97% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextbase_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextbase_ade20k.log)                |
| UperNet       | ConvNeXt-B-21k        | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 52.71% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextbase21k_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextbase21k_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextbase21k_ade20k.log)       |
| UperNet       | ConvNeXt-L-21k        | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 53.41% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextlarge21k_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextlarge21k_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextlarge21k_ade20k.log)    |
| UperNet       | ConvNeXt-XL-21k       | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 53.68% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextxlarge21k_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextxlarge21k_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_convnext/upernet_convnextxlarge21k_ade20k.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**