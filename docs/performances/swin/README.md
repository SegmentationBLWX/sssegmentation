# Introduction
```
@article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## ADE20k
| Model         | Backbone    | Crop Size  | Schedule                                | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                     | :-:             | :-:    | :-:                      |
| UperNet       | Swin-T      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 44.58% | [model]() &#124; [log]() |
| UperNet       | Swin-S      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 48.39% | [model]() &#124; [log]() |
| UperNet       | Swin-B      | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 51.02% | [model]() &#124; [log]() |