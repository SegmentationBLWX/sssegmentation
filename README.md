# Introduction
**SSSegmentation** is an open source strongly supervised semantic segmentation toolbox based on PyTorch.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.


# Documents
#### In English
https://sssegmentation.readthedocs.io/en/latest/


# Supported
#### Supported Backbones
- [UNet](https://arxiv.org/pdf/1505.04597.pdf)
- [CGNet](https://arxiv.org/pdf/1811.08201.pdf)
- [HRNet](https://arxiv.org/pdf/1908.07919.pdf)
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [ResNeSt](https://arxiv.org/pdf/2004.08955.pdf)
- [FastSCNN](https://arxiv.org/pdf/1902.04502.pdf)
- [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)
- [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
- [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf)
- [VisionTransformer](https://arxiv.org/pdf/2010.11929.pdf)
#### Supported Models
- [FCN](https://arxiv.org/pdf/1411.4038.pdf)
- [CE2P](https://arxiv.org/pdf/1809.05996.pdf)
- [SETR](https://arxiv.org/pdf/2012.15840.pdf)
- [ISNet](https://arxiv.org/pdf/2108.12382.pdf)
- [CCNet](https://arxiv.org/pdf/1811.11721.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983.pdf)
- [GCNet](https://arxiv.org/pdf/1904.11492.pdf)
- [DMNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf)
- [ISANet](https://arxiv.org/pdf/1907.12273.pdf)
- [EncNet](https://arxiv.org/pdf/1803.08904.pdf)
- [OCRNet](https://arxiv.org/pdf/1909.11065.pdf)
- [DNLNet](https://arxiv.org/pdf/2006.06668.pdf)
- [ANNNet](https://arxiv.org/pdf/1908.07678.pdf)
- [EMANet](https://arxiv.org/pdf/1907.13426.pdf)
- [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
- [PSANet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)
- [APCNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)
- [UPerNet](https://arxiv.org/pdf/1807.10221.pdf)
- [PointRend](https://arxiv.org/pdf/1912.08193.pdf)
- [Deeplabv3](https://arxiv.org/pdf/1706.05587.pdf)
- [Segformer](https://arxiv.org/pdf/2105.15203.pdf)
- [SemanticFPN](https://arxiv.org/pdf/1901.02446.pdf)
- [NonLocalNet](https://arxiv.org/pdf/1711.07971.pdf)
- [Deeplabv3Plus](https://arxiv.org/pdf/1802.02611.pdf)
- [MemoryNet-MCIBI](https://arxiv.org/pdf/2108.11819.pdf)
- [Mixed Precision (FP16) Training](https://arxiv.org/pdf/1710.03740.pdf)
#### Supported Datasets
- [LIP](http://sysu-hcp.net/lip/)
- [ATR](http://sysu-hcp.net/lip/overview.php)
- [HRF](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/)
- [CIHP](http://sysu-hcp.net/lip/overview.php)
- [DRIVE](https://drive.grand-challenge.org/)
- [STARE](http://cecas.clemson.edu/~ahoover/stare/)
- [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [MS COCO](https://cocodataset.org/#home)
- [MHPv1&v2](https://lv-mhp.github.io/dataset)
- [CHASE DB1](https://staffnet.kingston.ac.uk/~ku15565/)
- [CityScapes](https://www.cityscapes-dataset.com/)
- [Supervisely](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
- [SBUShadow](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCOStuff10k](https://cocodataset.org/#home)
- [Pascal Context](https://cs.stanford.edu/~roozbeh/pascal-context/)


# Citation
If you use this framework in your research, please cite this project:
```
@misc{ssseg2020,
    author = {Zhenchao Jin},
    title = {SSSegmentation: An Open Source Strongly Supervised Semantic Segmentation Toolbox Based on PyTorch},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/SegmentationBLWX/sssegmentation}},
}

@article{jin2021isnet,
  title={ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation},
  author={Jin, Zhenchao and Liu, Bin and Chu, Qi and Yu, Nenghai},
  journal={arXiv preprint arXiv:2108.12382},
  year={2021}
}

@article{jin2021mining,
  title={Mining Contextual Information Beyond Image for Semantic Segmentation},
  author={Jin, Zhenchao and Gong, Tao and Yu, Dongdong and Chu, Qi and Wang, Jian and Wang, Changhu and Shao, Jie},
  journal={arXiv preprint arXiv:2108.11819},
  year={2021}
}
```


# References
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)