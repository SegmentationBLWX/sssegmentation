# Introduction
```
@article{zhang2020resnest,
    title={ResNeSt: Split-Attention Networks},
    author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
    journal={arXiv preprint arXiv:2004.08955},
    year={2020}
}
All the reported models here are available at https://pan.baidu.com/s/1nPxHw5Px7a7jZMiX-ZxRhA (code is 3jvr)
```


# Results

## PASCAL VOC
| Model         | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | Rst-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | 77.41% | [model]() &#124; [log]() |
| PSPNet        | Rst-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| DeepLabV3     | Rst-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| DeepLabV3plus | Rst-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |

## ADE20k
| Model         | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | Rst-50-D8   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |
| PSPNet        | Rst-50-D16  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 46.03% | [model]() &#124; [log]() |
| DeepLabV3     | Rst-101-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 46.24% | [model]() &#124; [log]() |
| DeepLabV3plus | Rst-101-D16 | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |

## CityScapes
| Model         | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | Rst-50-D8   | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.14% | [model]() &#124; [log]() |
| PSPNet        | Rst-50-D16  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 78.70% | [model]() &#124; [log]() |
| DeepLabV3     | Rst-101-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 79.75% | [model]() &#124; [log]() |
| DeepLabV3plus | Rst-101-D16 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | 80.30% | [model]() &#124; [log]() |