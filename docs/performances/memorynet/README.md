# Introduction
```
@article{jin2021mining,
  title={Mining Contextual Information Beyond Image for Semantic Segmentation},
  author={Jin, Zhenchao and Gong, Tao and Yu, Dongdong and Chu, Qi and Wang, Jian and Wang, Changhu and Shao, Jie},
  journal={arXiv preprint arXiv:2108.11819},
  year={2021}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Performance

## COCOStuff-10k
| Model         | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU (ms+flip)  | Download                 |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| DeepLabV3     | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 38.84%/39.68%        | [model]() &#124; [log]() |
| DeepLabV3     | R-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 39.84%/41.49%        | [model]() &#124; [log]() |
| DeepLabV3     | S-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/32/150 | train/test      | 41.18%/42.15%        | [model]() &#124; [log]() |
| DeepLabV3     | HRNetV2p-W48 | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 39.77%/41.35%        | [model]() &#124; [log]() |
| DeepLabV3     | ViT-Large    | 512x512    | LR/POLICY/BS/EPOCH: 0.001/poly/16/110 | train/test      | 44.01%/45.23%        | [model]() &#124; [log]() |

## ADE20k
| Model         | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU (ms+flip)  | Download                 |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| DeepLabV3     | R-50-D8      | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 44.39%/45.95%        | [model]() &#124; [log]() |
| DeepLabV3     | R-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 45.66%/47.22%        | [model]() &#124; [log]() |
| DeepLabV3     | S-101-D8     | 512x512    | LR/POLICY/BS/EPOCH: 0.004/poly/16/180 | train/val       | 46.63%/47.36%        | [model]() &#124; [log]() |
| DeepLabV3     | HRNetV2p-W48 | 512x512    | LR/POLICY/BS/EPOCH: 0.004/poly/16/180 | train/val       | 45.79%/47.34%        | [model]() &#124; [log]() |
| DeepLabV3     | ViT-Large    | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130  | train/val       | 49.73%/50.99%        | [model]() &#124; [log]() |

## CityScapes
| Model         | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU (ms+flip)       | Download                 |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| DeepLabV3     | R-50-D8      | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | trainval/test   | 79.90%               | [model]() &#124; [log]() |
| DeepLabV3     | R-101-D8     | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/440  | trainval/test   | 82.03%               | [model]() &#124; [log]() |
| DeepLabV3     | S-101-D8     | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/500  | trainval/test   | 81.59%               | [model]() &#124; [log]() |
| DeepLabV3     | HRNetV2p-W48 | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/500  | trainval/test   | 82.55%               | [model]() &#124; [log]() |

## LIP
| Model         | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU/mIoU (flip)     | Download                 |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:                  | :-:                      |
| DeepLabV3     | R-50-D8      | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150  | train/val       | 53.73%/54.08%        | [model]() &#124; [log]() |
| DeepLabV3     | R-101-D8     | 473x473    | LR/POLICY/BS/EPOCH: 0.01/poly/32/150  | train/val       | 55.02%/55.42%        | [model]() &#124; [log]() |
| DeepLabV3     | S-101-D8     | 473x473    | LR/POLICY/BS/EPOCH: 0.007/poly/40/150 | train/val       | 56.21%/56.34%        | [model]() &#124; [log]() |
| DeepLabV3     | HRNetV2p-W48 | 473x473    | LR/POLICY/BS/EPOCH: 0.007/poly/40/150 | train/val       | 56.40%/56.99%        | [model]() &#124; [log]() |