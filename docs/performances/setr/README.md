# Introduction
```
@inproceedings{zheng2021rethinking,
    title={Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers},
    author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip HS and others},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={6881--6890},
    year={2021}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## ADE20k
| Model         | Backbone    | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:         | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| Naive         | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 48.43% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrnaive_vitlarge_ade20k_train.log) |
| PUP           | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 48.51% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrpup_vitlarge_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrpup_vitlarge_ade20k_train.log) |
| MLA           | ViT-Large   | 512x512    | LR/POLICY/BS/EPOCH: 0.01/poly/16/130 | train/val       | 49.61% | [model](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrmla_vitlarge_ade20k_train.pth) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_setr/setrmla_vitlarge_ade20k_train.log) |