## Introduction

<a href="https://github.com/microsoft/Swin-Transformer">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/swin.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2103.14030.pdf">Swin Transformer (ICCV'2021)</a></summary>

```latex
@article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
}
```

</details>


## Results

#### ADE20k
| Segmentor     | Backbone    | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                    |
| :-:           | :-:         | :-:        | :-:                                     | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                         |
| UperNet       | Swin-T      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 44.58% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_swintiny_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_swin/upernet_swintiny_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_swin/upernet_swintiny_ade20k_train.log)    |
| UperNet       | Swin-S      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.39% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_swinsmall_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_swin/upernet_swinsmall_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_swin/upernet_swinsmall_ade20k_train.log) |
| UperNet       | Swin-B      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 51.02% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_swinbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_swin/upernet_swinbase_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_swin/upernet_swinbase_ade20k_train.log)    |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**