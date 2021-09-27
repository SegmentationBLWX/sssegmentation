# Introduction
```
@article{poudel2019fast,
    title={Fast-scnn: Fast semantic segmentation network},
    author={Poudel, Rudra PK and Liwicki, Stephan and Cipolla, Roberto},
    journal={arXiv preprint arXiv:1902.04502},
    year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## CityScapes
| Model                  | Backbone     | Crop Size  | Schedule                              | Train/Eval Set  | mIoU   | Download                 |
| :-:                    | :-:          | :-:        | :-:                                   | :-:             | :-:    | :-:                      |
| DepthwiseSeparableFCN  | FastSCNN     | 512x1024   | LR/POLICY/BS/EPOCH: 0.12/poly/32/1750 | train/val       | 71.53% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastscnn/depthwiseseparablefcn_fastscnn_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastscnn/depthwiseseparablefcn_fastscnn_cityscapes_train.log) |