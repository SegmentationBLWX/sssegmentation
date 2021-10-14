# Introduction
```
@inproceedings{yu2018bisenet,
    title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
    author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    pages={325--341},
    year={2018}
}
All the reported models here are available at https://pan.baidu.com/s/1gD-NJJWOtaHCtB0qHE79rA (code is s757).
```


# Results

## CityScapes
| Model         | Backbone             | Crop Size  | Schedule                             | Train/Eval Set  | mIoU   | Download                 |
| :-:           | :-:                  | :-:        | :-:                                  | :-:             | :-:    | :-:                      |
| FCN           | BiSeNetV1, R-18-D32  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | 75.76% | [model]() &#124; [log]() |
| FCN           | BiSeNetV1, R-50-D32  | 1024x1024  | LR/POLICY/BS/EPOCH: 0.05/poly/16/860 | train/val       | -      | [model]() &#124; [log]() |