# Introduction

<a href="https://github.com/facebookresearch/ConvNeXt">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/backbones">Code Snippet</a>

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


# Results

## ADE20k
| Model         | Backbone              | Crop Size  | Schedule                               | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                         |
| :-:           | :-:                   | :-:        | :-:                                    | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                              |
| UperNet       | ConvNeXt-T-fp16       | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | -      | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnexttinyfp16_ade20k.py) &#124; [model]() &#124; [log]()       |
| UperNet       | ConvNeXt-S-fp16       | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | -      | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextsmallfp16_ade20k.py) &#124; [model]() &#124; [log]()          |
| UperNet       | ConvNeXt-B-fp16       | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | -      | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextbasefp16_ade20k.py) &#124; [model]() &#124; [log]()       |
| UperNet       | ConvNeXt-B-21k-fp16   | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | -      | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextbase21kfp16_ade20k.py) &#124; [model]() &#124; [log]() |
| UperNet       | ConvNeXt-L-21k-fp16   | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | -      | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextlarge21kfp16_ade20k.py) &#124; [model]() &#124; [log]()    |
| UperNet       | ConvNeXt-XL-21k-fp16  | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | -      | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/convnext/upernet_convnextxlarge21kfp16_ade20k.py) &#124; [model]() &#124; [log]() |


# More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**