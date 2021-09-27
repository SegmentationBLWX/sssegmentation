# Introduction
```
@article{yu2021bisenet,
    title={Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation},
    author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
    journal={International Journal of Computer Vision},
    pages={1--18},
    year={2021},
    publisher={Springer}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## CityScapes
| Model         | Backbone        | Crop Size  | Schedule                              | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:             | :-:        | :-:                                   | :-:             | :-:    | :-:                      |
| FCN           | BiSeNetV2       | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/32/1720 | train/val       | -      | [model]() &#124; [log]() |
| FCN           | BiSeNetV2-FP16  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/32/1720 | train/val       | -      | [model]() &#124; [log]() |