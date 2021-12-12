# Introduction

<a href="https://github.com/facebookresearch/MaskFormer">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/maskformer">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2107.06278.pdf">MaskFormer (NeurIPS'2021)</a></summary>

```latex
@inproceedings{cheng2021per,
    title={Per-pixel classification is not all you need for semantic segmentation},
    author={Cheng, Bowen and Schwing, Alex and Kirillov, Alexander},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021}
}
```

</details>


# Results

## ADE20k
| Model         | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| UperNet       | Swin-Tiny   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 47.31% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_maskformer/maskformer_swintiny_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_maskformer/maskformer_swintiny_ade20k_train.log) |
| UperNet       | Swin-Small  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       |        | [model]() &#124; [log]() |
| UperNet       | Swin-Large  | 640x640    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 53.22% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_maskformer/maskformer_swinbase_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_maskformer/maskformer_swinbase_ade20k_train.log) |


# More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**