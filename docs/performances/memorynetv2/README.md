## Introduction

<a href="https://github.com/SegmentationBLWX/sssegmentation">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/memorynetv2/memorynetv2.py">Code Snippet</a>

<details>
<summary align="left"><a href="">MemoryNetV2-MCIBI++ (TPAMI'2022)</a></summary>

```latex
```

</details>


## Results of Different Frameworks

#### LIP
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU             | Download                                  |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:              | :-:                                       |
| FCN           | R-50-D8      | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]()   |
| PSNet         | R-50-D8      | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]()   |
| UperNet       | R-50-D8      | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]()   |
| DeepLabV3     | R-50-D8      | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]()   |

#### ADE20k
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU             | Download                                |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:              | :-:                                     |
| FCN           | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |
| PSNet         | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |
| DeepLabV3     | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |

#### CityScapes
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU             | Download                                |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:              | :-:                                     |
| FCN           | R-50-D8      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/220    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |
| PSNet         | R-50-D8      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/220    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | R-50-D8      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/220    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |
| DeepLabV3     | R-50-D8      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/220    | train/val       |                  | [cfg]() &#124; [model]() &#124; [log]() |

#### COCOStuff-10k
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU             | Download                                |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:              | :-:                                     |
| FCN           | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110   | train/test      |                  | [cfg]() &#124; [model]() &#124; [log]() |
| PSNet         | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110   | train/test      |                  | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110   | train/test      |                  | [cfg]() &#124; [model]() &#124; [log]() |
| DeepLabV3     | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110   | train/test      |                  | [cfg]() &#124; [model]() &#124; [log]() |


## SOTA Results

#### VSPW
| Segmentor     | Backbone     | Crop Size  | Schedule                                   | Train/Eval Set  | mIoU                            | Download                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:           | :-:          | :-:        | :-:                                        | :-:             | :-:                             | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| UperNet       | R-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/240       | train/val       | 43.21%                          | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnet101os8_vspw.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_vspw.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_vspw.log) |
| UperNet       | Swin-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/240    | train/val       | 56.04%                          | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_swinlarge_vspw.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_swinlarge_vspw.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_swinlarge_vspw.log)          |

#### PASCAL-VOC
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU                            | Download                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:                             | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| UperNet       | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60     | train/val       | 79.48                           | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnet50os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet50os8_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet50os8_voc.log)    |
| UperNet       | R-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/60     | train/val       | 80.42                           | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnet101os8_voc.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_voc.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_voc.pth) |

#### PASCAL-Context
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU/mIoU (ms+flip)             | Download                 |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:                             | :-:                      |
| UperNet       | R-101-D8     | 480x480    | LR/POLICY/BS/EPOCH: 0.004/poly/16/260   | train/val       | 55.63%/56.82%                   | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | S-101-D8     | 480x480    | LR/POLICY/BS/EPOCH: 0.004/poly/16/260   | train/val       | 56.83%/57.92%                   | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | Swin-Large   | 480x480    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/260 | train/val       | 62.37%/64.01%                   | [cfg]() &#124; [model]() &#124; [log]() |

#### LIP
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU/mIoU (flip)/mIoU (ms+flip) | Download                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:                             | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| UperNet       | R-101-D8     | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       | 55.87%/56.26%/56.32%            | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnet101os8_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_lip.log)    |
| UperNet       | S-101-D8     | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150    | train/val       | 56.58%/56.77%/57.08%            | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnest101os8_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnest101os8_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnest101os8_lip.log) |
| UperNet       | Swin-Large   | 473x473    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/110 | train/val       | 59.58%/59.89%/59.91%            | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_swinlarge_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_swinlarge_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_swinlarge_lip.log)             |
| DeepLabV3     | HRNetV2p-W48 | 512x512    | LR/POLICY/BS/EPOCH: 0.007/poly/40/150   | train/val       | 56.70%/57.27%/57.42%            | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_aspp_hrnetv2w48_lip.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_aspp_hrnetv2w48_lip.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_aspp_hrnetv2w48_lip.log)                   |

#### ADE20k
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU/mIoU (ms+flip)             | Download                 |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:                             | :-:                      |
| UperNet       | R-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130    | train/val       | 46.38%/47.93%                   | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | S-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.004/poly/16/180   | train/val       | 47.59%/48.56%                   | [cfg]() &#124; [model]() &#124; [log]() |
| UperNet       | Swin-Large   | 640x640    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/130 | train/val       | 53.48%/54.50%                   | [cfg]() &#124; [model]() &#124; [log]() |

#### CityScapes
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU (ms+flip)                  | Download                 |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:                             | :-:                      |
| DeepLabV3     | R-101-D8     | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440    | trainval/test   |                                 | [cfg]() &#124; [model]() &#124; [log]() |
| DeepLabV3     | S-101-D8     | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440    | trainval/test   | 81.70%                          | [cfg]() &#124; [model]() &#124; [log]() |
| DeepLabV3     | HRNetV2p-W48 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440    | trainval/test   | 82.74%                          | [cfg]() &#124; model]() &#124; [log]() |

#### COCOStuff-10k
| Segmentor     | Backbone     | Crop Size  | Schedule                                | Train/Eval Set  | mIoU/mIoU (ms+flip)             | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| :-:           | :-:          | :-:        | :-:                                     | :-:             | :-:                             | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| UperNet       | R-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110   | train/test      | 40.41%/41.84%                   | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnet101os8_cocostuff10k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_cocostuff10k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnet101os8_cocostuff10k.log)    |
| UperNet       | S-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/32/150   | train/test      | 41.81%/42.71%                   | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_resnest101os8_cocostuff10k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnest101os8_cocostuff10k.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_resnest101os8_cocostuff10k.log) |
| UperNet       | Swin-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.00006/poly/16/150 | train/test      | 49.11%/50.27%                   | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/memorynetv2/memorynetv2_upernet_swinlarge_cocostuff10k.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_swinlarge_vspw.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_memorynetv2/memorynetv2_upernet_swinlarge_cocostuff10k.log)                     |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**