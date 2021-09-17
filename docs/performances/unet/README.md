# Introduction
```
@inproceedings{ronneberger2015u,
    title={U-net: Convolutional networks for biomedical image segmentation},
    author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
    booktitle={International Conference on Medical image computing and computer-assisted intervention},
    pages={234--241},
    year={2015},
    organization={Springer}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## HRF
| Model         | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                 |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | UNet-S5-D16  | 256x256    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 79.88% | [model]() &#124; [log]() |
| PSPNet        | UNet-S5-D16  | 256x256    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.26% | [model]() &#124; [log]() |
| DeepLabV3     | UNet-S5-D16  | 256x256    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | -      | [model]() &#124; [log]() |

## DRIVE
| Model         | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                 |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | UNet-S5-D16  | 64x64      | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | -      | [model]() &#124; [log]() |
| PSPNet        | UNet-S5-D16  | 64x64      | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | -      | [model]() &#124; [log]() |
| DeepLabV3     | UNet-S5-D16  | 64x64      | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | -      | [model]() &#124; [log]() |

## STARE
| Model         | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                 |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | -      | [model]() &#124; [log]() |
| PSPNet        | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | -      | [model]() &#124; [log]() |
| DeepLabV3     | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 81.03% | [model]() &#124; [log]() |

## CHASE DB1
| Model         | Backbone     | Crop Size  | Schedule                             | Train/Eval Set  | Dice   | Download                 |
| :-:           | :-:          | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.50% | [model]() &#124; [log]() |
| PSPNet        | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.50% | [model]() &#124; [log]() |
| DeepLabV3     | UNet-S5-D16  | 128x128    | LR/POLICY/BS/EPOCH: 0.01/poly/16/1   | train/val       | 80.54% | [model]() &#124; [log]() |