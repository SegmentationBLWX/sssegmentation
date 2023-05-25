## Introduction

<a href="https://github.com/Eromera/erfnet_pytorch">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/erfnet.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://ieeexplore.ieee.org/document/8063438">ERFNet (T-ITS'2017)</a></summary>

```latex
@article{romera2017erfnet,
    title={Erfnet: Efficient residual factorized convnet for real-time semantic segmentation},
    author={Romera, Eduardo and Alvarez, Jos{\'e} M and Bergasa, Luis M and Arroyo, Roberto},
    journal={IEEE Transactions on Intelligent Transportation Systems},
    volume={19},
    number={1},
    pages={263--272},
    year={2017},
    publisher={IEEE}
}
```

</details>


## Results

#### CityScapes
| Segmentor     | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                           |
| :-:           | :-:          | :-:        | :-:                                   | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                |
| FCN           | ERFNet       | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/16/860  | train/val       | 76.44% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fcn/fcn_erfnet_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_erfnet/fcn_erfnet_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_erfnet/fcn_erfnet_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**