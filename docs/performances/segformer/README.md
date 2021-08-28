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
| MIT-B0      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 37.57% | [model]() &#124; [log]() |
| MIT-B1      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 42.25% | [model]() &#124; [log]() |
| MIT-B2      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 46.35% | [model]() &#124; [log]() |
| MIT-B3      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.31% | [model]() &#124; [log]() |
| MIT-B4      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.59% | [model]() &#124; [log]() |
| MIT-B5      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 49.61% | [model]() &#124; [log]() |