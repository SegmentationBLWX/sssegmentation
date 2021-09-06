# Introduction
```
@article{xie2021segformer,
    title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
    author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
    journal={arXiv preprint arXiv:2105.15203},
    year={2021}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## ADE20k
| Backbone    | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                 |
| :-:         | :-:        | :-:                                     | :-:             | :-:    | :-:                      |
| MIT-B0      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 37.57% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb0_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb0_ade20k_train.log) |
| MIT-B1      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 42.25% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb1_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb1_ade20k_train.log) |
| MIT-B2      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 46.35% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb2_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb2_ade20k_train.log) |
| MIT-B3      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.31% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb3_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb3_ade20k_train.log) |
| MIT-B4      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.59% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb4_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb4_ade20k_train.log) |
| MIT-B5      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 49.61% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb5_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_segformer/segformer_mitb5_ade20k_train.log) |