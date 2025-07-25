## Introduction

<a href="https://github.com/fudan-zvg/SETR">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/setr/setr.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2012.15840.pdf">SETR (CVPR'2021)</a></summary>

```latex
@inproceedings{zheng2021rethinking,
    title={Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers},
    author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip HS and others},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={6881--6890},
    year={2021}
}
```

</details>


## Results

#### PASCAL VOC

| Segmentor     | Pretrain               | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                               |
| :-:           | :-:                    | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                    |
| Naive         | ImageNet-22k-384x384   | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 84.23% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/setr/setrnaive_vitlarge_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_voc.log) |

#### ADE20k

| Segmentor     | Pretrain               | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                        |
| :-:           | :-:                    | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                             |
| Naive         | ImageNet-22k-384x384   | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 48.43% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/setr/setrnaive_vitlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_ade20k.log) |
| PUP           | ImageNet-22k-384x384   | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 48.51% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/setr/setrpup_vitlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrpup_vitlarge_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrpup_vitlarge_ade20k.log)       |
| MLA           | ImageNet-22k-384x384   | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 49.61% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/setr/setrmla_vitlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrmla_vitlarge_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrmla_vitlarge_ade20k.log)       |

#### CityScapes

| Segmentor     | Pretrain               | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                    |
| :-:           | :-:                    | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                         |
| Naive         | ImageNet-22k-384x384   | ViT-Large   | 768x768    | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/setr/setrnaive_vitlarge_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_cityscapes.log) |


## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**