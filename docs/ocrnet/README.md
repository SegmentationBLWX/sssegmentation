# Introduction
```
@article{yuan2019object,
    title={Object-contextual representations for semantic segmentation},
    author={Yuan, Yuhui and Chen, Xilin and Wang, Jingdong},
    journal={arXiv preprint arXiv:1909.11065},
    year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1nPxHw5Px7a7jZMiX-ZxRhA (code is 3jvr)
```


# Results

## PASCAL VOC
| Backbone           | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:                | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| HRNetV2p-W18-Small | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| HRNetV2p-W18       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| HRNetV2p-W48       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |

## ADE20k
| Backbone           | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:                | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| HRNetV2p-W18-Small | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |
| HRNetV2p-W18       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |
| HRNetV2p-W48       | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |

## CityScapes
| Backbone           | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:                | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| HRNetV2p-W18-Small | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | -      | [model]() &#124; [log]() |
| HRNetV2p-W18       | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | -      | [model]() &#124; [log]() |
| HRNetV2p-W48       | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/440  | train/val       | -      | [model]() &#124; [log]() |