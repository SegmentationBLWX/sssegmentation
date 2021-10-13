# Introduction
```
@article{wu2019fastfcn,
    title={Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation},
    author={Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu, Yizhou},
    journal={arXiv preprint arXiv:1903.11816},
    year={2019}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## CityScapes
| Model         | Backbone  | Crop Size  | Schedule                            | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:       | :-:        | :-:                                 | :-:             | :-:    | :-:                      |
| EncNet        | R-50-D32  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220 | train/val       | 78.42% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_encnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_encnet_resnet50os8_cityscapes_train.log) |
| PSPNet        | R-50-D32  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220 | train/val       | 79.36% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_pspnet_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_pspnet_resnet50os8_cityscapes_train.log) |
| DeepLabV3     | R-50-D32  | 512x1024   | LR/POLICY/BS/EPOCH: 0.01/poly/8/220 | train/val       | 79.96% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_deeplabv3_resnet50os8_cityscapes_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_fastfcn/fastfcn_deeplabv3_resnet50os8_cityscapes_train.log) |