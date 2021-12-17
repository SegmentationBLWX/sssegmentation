# Introduction

<a href="https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/backbones">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2104.13840.pdf">Twins (NeurIPS'2021)</a></summary>

```latex
@article{chu2021twins,
    title={Twins: Revisiting spatial attention design in vision transformers},
    author={Chu, Xiangxiang and Tian, Zhi and Wang, Yuqing and Zhang, Bo and Ren, Haibing and Wei, Xiaolin and Xia, Huaxia and Shen, Chunhua},
    journal={arXiv preprint arXiv:2104.13840},
    year={2021}altgvt
}
```

</details>


# Results

## ADE20k
| Model         | Backbone    | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                     | :-:             | :-:    | :-:                      |
| UperNet       | PCPVT-S     | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       |        | [model]() &#124; [log]() |
| UperNet       | PCPVT-B     | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       |        | [model]() &#124; [log]() |
| UperNet       | PCPVT-L     | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       |        | [model]() &#124; [log]() |
| UperNet       | SVT-S       | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       |        | [model]() &#124; [log]() |
| UperNet       | SVT-B       | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       |        | [model]() &#124; [log]() |
| UperNet       | SVT-L       | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       |        | [model]() &#124; [log]() |


# More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**