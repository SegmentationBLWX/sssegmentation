## Introduction

<a href="https://github.com/facebookresearch/mae">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/mae.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2111.06377.pdf">MAE (CVPR'2022)</a></summary>

```latex
@inproceedings{he2022masked,
    title={Masked autoencoders are scalable vision learners},
    author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={16000--16009},
    year={2022}
}
```

</details>


## Results

#### ADE20k
| Segmentor     | Backbone              | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                      |
| :-:           | :-:                   | :-:        | :-:                                  | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                           |
| UperNet       | MAE-Vit-B             | 512x512    | LR/POLICY/BS/EPOCH: 1e-4/poly/16/130 | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mae/upernet_maevitbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mae/upernet_maevitbase_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mae/upernet_maevitbase_ade20k.log)  |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**