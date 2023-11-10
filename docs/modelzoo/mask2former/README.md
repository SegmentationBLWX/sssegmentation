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

| Segmentor      | Pretrain               | Backbone            | Crop Size  | Schedule                               | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                      |
| :-:            | :-:                    | :-:                 | :-:        | :-:                                    | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                           |
| Mask2Former    | ImageNet-1k-224x224    | Swin-T-PYTORCHFP16  | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 48.93% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swintiny_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swintiny_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swintiny_ade20k.log)    |
| Mask2Former    | ImageNet-1k-224x224    | Swin-S-PYTORCHFP16  | 512x512    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinsmall_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinsmall_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinsmall_ade20k.log) |
| Mask2Former    | ImageNet-22k-384x384   | Swin-B-PYTORCHFP16  | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       | 54.04% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinbase_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinbase_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinbase_ade20k.log)    |
| Mask2Former    | ImageNet-22k-384x384   | Swin-L-PYTORCHFP16  | 640x640    | LR/POLICY/BS/EPOCH: 0.0001/poly/16/130 | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinlarge_ade20k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinlarge_ade20k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinlarge_ade20k.log) |

#### Cityscapes

| Segmentor      | Pretrain               | Backbone             | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :-:            | :-:                    | :-:                  | :-:        | :-:                                     | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Mask2Former    | ImageNet-1k-224x224    | Swin-T-PYTORCHFP16   | 512x1024   | LR/POLICY/BS/EPOCH: 0.0001/poly/16/500  | train/val       | 81.85% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swintiny_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swintiny_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swintiny_cityscapes.log)    |
| Mask2Former    | ImageNet-1k-224x224    | Swin-S-PYTORCHFP16   | 512x1024   | LR/POLICY/BS/EPOCH: 0.0001/poly/16/500  | train/val       | 82.91% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinsmall_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinsmall_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinsmall_cityscapes.log) |
| Mask2Former    | ImageNet-22k-384x384   | Swin-B-PYTORCHFP16   | 512x1024   | LR/POLICY/BS/EPOCH: 0.0001/poly/16/500  | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinbase_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinbase_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinbase_cityscapes.log)    |
| Mask2Former    | ImageNet-22k-384x384   | Swin-L-PYTORCHFP16   | 512x1024   | LR/POLICY/BS/EPOCH: 0.0001/poly/16/500  | train/val       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/mask2former/mask2former_swinlarge_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinlarge_cityscapes.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_mask2former/mask2former_swinlarge_cityscapes.log) |

## More

You can also download the model weights from following sources:

- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**