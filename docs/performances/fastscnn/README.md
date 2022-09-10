## Introduction

<a href="">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/fastscnn.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1902.04502.pdf">Fast-SCNN (ArXiv'2019)</a></summary>

```latex
@article{poudel2019fast,
    title={Fast-scnn: Fast semantic segmentation network},
    author={Poudel, Rudra PK and Liwicki, Stephan and Cipolla, Roberto},
    journal={arXiv preprint arXiv:1902.04502},
    year={2019}
}
```

</details>


## Results

#### CityScapes
| Segmentor              | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| :-:                    | :-:          | :-:        | :-:                                   | :-:             | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| DepthwiseSeparableFCN  | FastSCNN     | 512x1024   | LR/POLICY/BS/EPOCH: 0.12/poly/32/1750 | train/val       | 71.53% | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/sssegmentation/main/ssseg/configs/fastscnn/depthwiseseparablefcn_fastscnn_cityscapes.py) &#124; [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastscnn/depthwiseseparablefcn_fastscnn_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastscnn/depthwiseseparablefcn_fastscnn_cityscapes_train.log) |


## More
You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA with access code **s757**