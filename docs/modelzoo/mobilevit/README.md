## Introduction

<a href="https://github.com/apple/ml-cvnets">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/mobilevit.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2110.02178.pdf">MobileViT (ICLR'2022)</a></summary>

```latex
@article{mehta2021mobilevit,
    title={Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer},
    author={Mehta, Sachin and Rastegari, Mohammad},
    journal={arXiv preprint arXiv:2110.02178},
    year={2021}
}
```

</details>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2206.02680.pdf">MobileViTV2 (ArXiv'2022)</a></summary>

```latex
@article{mehta2022separable,
    title={Separable self-attention for mobile vision transformers},
    author={Mehta, Sachin and Rastegari, Mohammad},
    journal={arXiv preprint arXiv:2206.02680},
    year={2022}
}
```

</details>


## Results

#### PASCAL VOC + COCO VOC Subset

| Segmentor     | Pretrain               | Backbone         | Crop Size  | Schedule                                 | Train/Eval Set                                 | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :-:           | :-:                    | :-:              | :-:        | :-:                                      | :-:                                            | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                             |
| DeepLabV3     | ImageNet-1k-224x224    | MobileViT-Small  | 512x512    | LR/POLICY/BS/EPOCH: 0.0009/cosine/64/50  | voc trainaug + cocovocsubet train / voc val    |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mobilevit/deeplabv3_mobilevits_voc-cocosubvoc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilevit/deeplabv3_mobilevits_voc-cocosubvoc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mobilevit/deeplabv3_mobilevits_voc-cocosubvoc.pth)    |


## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**