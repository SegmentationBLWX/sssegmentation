## Introduction

<a href="https://github.com/facebookresearch/Mask2Former">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/mask2former/mask2former.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2112.01527.pdf">Mask2Former (CVPR'2022)</a></summary>

```latex
@inproceedings{cheng2021mask2former,
    title={Masked-attention Mask Transformer for Universal Image Segmentation},
    author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
    journal={CVPR},
    year={2022}
}
```

</details>


## Results

#### ADE20k
| Segmentor      | Pretrain               | Backbone    | Crop Size  | Schedule                               | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :-:            | :-:                    | :-:         | :-:        | :-:                                    | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Mask2Former    | ImageNet-1k-224x224    | Swin-Tiny   | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swintiny_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swintiny_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swintiny_ade20k_train.log)    |
| Mask2Former    | ImageNet-1k-224x224    | Swin-Small  | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinsmall_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinsmall_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinsmall_ade20k_train.log) |
| Mask2Former    | ImageNet-22k-384x384   | Swin-Base   | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 54.04% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinbase_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinbase_ade20k_train.log)    |

#### Cityscapes
| Segmentor      | Pretrain               | Backbone    | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :-:            | :-:                    | :-:         | :-:        | :-:                                     | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                       |

## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**