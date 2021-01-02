# Introduction
```
@inproceedings{sandler2018mobilenetv2,
    title={Mobilenetv2: Inverted residuals and linear bottlenecks},
    author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={4510--4520},
    year={2018}
}
@inproceedings{Howard_2019_ICCV,
    title={Searching for MobileNetV3},
    author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    pages={1314-1324},
    month={October},
    year={2019},
    doi={10.1109/ICCV.2019.00140}}
}
All the reported models here are available at https://pan.baidu.com/s/1nPxHw5Px7a7jZMiX-ZxRhA (code is 3jvr)
```


# Results

## PASCAL VOC
| Model         | Backbone | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:      | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| PSPNet        | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| DeepLabV3     | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |
| DeepLabV3plus | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60  | trainaug/val    | -      | [model]() &#124; [log]() |

## ADE20k
| Model         | Backbone | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:      | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |
| PSPNet        | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |
| DeepLabV3     | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |
| DeepLabV3plus | M-V2-D8  | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | -      | [model]() &#124; [log]() |

## CityScapes
| Model         | Backbone | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:      | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | [model]() &#124; [log]() |
| PSPNet        | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | [model]() &#124; [log]() |
| DeepLabV3     | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | [model]() &#124; [log]() |
| DeepLabV3plus | M-V2-D8  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220  | train/val       | -      | [model]() &#124; [log]() |