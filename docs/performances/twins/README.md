## Introduction

<a href="https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/twins.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2104.13840.pdf">Twins (NeurIPS'2021)</a></summary>

```latex
@article{chu2021twins,
    title={Twins: Revisiting spatial attention design in vision transformers},
    author={Chu, Xiangxiang and Tian, Zhi and Wang, Yuqing and Zhang, Bo and Ren, Haibing and Wei, Xiaolin and Xia, Huaxia and Shen, Chunhua},
    journal={arXiv preprint arXiv:2104.13840},
    year={2021}
}
```

</details>


## Results

#### ADE20k
| Segmentor     | Backbone    | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                         |
| :-:           | :-:         | :-:        | :-:                                     | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                              |
| UperNet       | SVT-S       | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 46.16% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_svtsmall_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_svtsmall_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_svtsmall_ade20k_train.log)       |
| UperNet       | SVT-B       | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.05% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_svtbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_svtbase_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_svtbase_ade20k_train.log)          |
| UperNet       | SVT-L       | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 49.80% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_svtlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_svtlarge_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_svtlarge_ade20k_train.log)       |
| UperNet       | PCPVT-S     | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 46.07% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_pcpvtsmall_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_pcpvtsmall_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_pcpvtsmall_ade20k_train.log) |
| UperNet       | PCPVT-B     | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.06% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_pcpvtbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_pcpvtbase_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_pcpvtbase_ade20k_train.log)    |
| UperNet       | PCPVT-L     | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 49.35% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/upernet/upernet_pcpvtlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_pcpvtlarge_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_twins/upernet_pcpvtlarge_ade20k_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**