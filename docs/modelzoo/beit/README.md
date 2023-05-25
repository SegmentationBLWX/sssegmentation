## Introduction

<a href="https://github.com/microsoft/unilm/tree/master/beit">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/beit.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2106.08254.pdf">BEiT (ICLR'2022)</a></summary>

```latex
@article{bao2021beit,
    title={Beit: Bert pre-training of image transformers},
    author={Bao, Hangbo and Dong, Li and Wei, Furu},
    journal={arXiv preprint arXiv:2106.08254},
    year={2021}
}
```

</details>


## Results

#### ADE20k
| Segmentor     | Backbone              | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                      |
| :-:           | :-:                   | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                           |
| UperNet       | BEiT-B                | 640x640    | LR/POLICY/BS/EPOCH: 3e-5/poly/16/130 | train/val       | 53.22% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/beit/upernet_beitbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_beit/upernet_beitbase_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_beit/upernet_beitbase_ade20k.log)     |
| UperNet       | BEiT-L                | 640x640    | LR/POLICY/BS/EPOCH: 3e-5/poly/16/130 | train/val       | 56.52% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/beit/upernet_beitlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_beit/upernet_beitlarge_ade20k.zip) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_beit/upernet_beitlarge_ade20k.log)  |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**