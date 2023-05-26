<div align="center">
  <img src="./docs/logo.png" width="600"/>
</div>
<br />

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://sssegmentation.readthedocs.io/en/latest/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sssegmentation)](https://pypi.org/project/sssegmentation/)
[![PyPI](https://img.shields.io/pypi/v/sssegmentation)](https://pypi.org/project/sssegmentation)
[![license](https://img.shields.io/github/license/SegmentationBLWX/sssegmentation.svg)](https://github.com/SegmentationBLWX/sssegmentation/blob/master/LICENSE)
[![PyPI - Downloads](https://pepy.tech/badge/sssegmentation)](https://pypi.org/project/sssegmentation/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sssegmentation?style=flat-square)](https://pypi.org/project/sssegmentation/)
[![issue resolution](https://isitmaintained.com/badge/resolution/SegmentationBLWX/sssegmentation.svg)](https://github.com/SegmentationBLWX/sssegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/SegmentationBLWX/sssegmentation.svg)](https://github.com/SegmentationBLWX/sssegmentation/issues)

Documents: https://sssegmentation.readthedocs.io/en/latest/


## Introduction

SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.


## Major Features

- **High Performance**

  The performance of re-implemented segmentation algorithms is better than or comparable to other codebases.
 
- **Modular Design and Unified Benchmark**
  
  Various segmentation methods are unified into several specific modules.
  Benefiting from this design, SSSegmentation can integrate a great deal of popular and contemporary semantic segmentation frameworks and then, train and test them on unified benchmarks.
 

## Benchmark and Model Zoo

#### Supported Backbones

| Backbone               | Model Zoo                                    | Paper Link                                                    | Code Snippet                                             |
| :-:                    | :-:                                          | :-:                                                           | :-:                                                      |
| UNet                   | [Click](./docs/modelzoo/unet)                | [MICCAI 2015](https://arxiv.org/pdf/1505.04597.pdf)           | [Click](./ssseg/modules/models/backbones/unet.py)        |
| BEiT                   | [Click](./docs/modelzoo/beit)                | [ICLR 2022](https://arxiv.org/pdf/2106.08254.pdf)             | [Click](./ssseg/modules/models/backbones/beit.py)        |
| Twins                  | [Click](./docs/modelzoo/twins)               | [NeurIPS 2021](https://arxiv.org/pdf/2104.13840.pdf)          | [Click](./ssseg/modules/models/backbones/twins.py)       |
| CGNet                  | [Click](./docs/modelzoo/cgnet)               | [TIP 2020](https://arxiv.org/pdf/1811.08201.pdf)              | [Click](./ssseg/modules/models/backbones/cgnet.py)       |
| HRNet                  | [Click](./docs/modelzoo/ocrnet)              | [CVPR 2019](https://arxiv.org/pdf/1908.07919.pdf)             | [Click](./ssseg/modules/models/backbones/hrnet.py)       |
| ERFNet                 | [Click](./docs/modelzoo/erfnet)              | [T-ITS 2017](https://ieeexplore.ieee.org/document/8063438)    | [Click](./ssseg/modules/models/backbones/erfnet.py)      |
| ResNet                 | [Click](./docs/modelzoo/fcn)                 | [CVPR 2016](https://arxiv.org/pdf/1512.03385.pdf)             | [Click](./ssseg/modules/models/backbones/resnet.py)      |
| ResNeSt                | [Click](./docs/modelzoo/resnest)             | [ArXiv 2020](https://arxiv.org/pdf/2004.08955.pdf)            | [Click](./ssseg/modules/models/backbones/resnest.py)     |
| ConvNeXt               | [Click](./docs/modelzoo/convnext)            | [CVPR 2022](https://arxiv.org/pdf/2201.03545.pdf)             | [Click](./ssseg/modules/models/backbones/convnext.py)    |
| FastSCNN               | [Click](./docs/modelzoo/fastscnn)            | [ArXiv 2019](https://arxiv.org/pdf/1902.04502.pdf)            | [Click](./ssseg/modules/models/backbones/fastscnn.py)    |
| BiSeNetV1              | [Click](./docs/modelzoo/bisenetv1)           | [ECCV 2018](https://arxiv.org/pdf/1808.00897.pdf)             | [Click](./ssseg/modules/models/backbones/bisenetv1.py)   |
| BiSeNetV2              | [Click](./docs/modelzoo/bisenetv2)           | [IJCV 2021](https://arxiv.org/pdf/2004.02147.pdf)             | [Click](./ssseg/modules/models/backbones/bisenetv2.py)   |
| MobileNetV2            | [Click](./docs/modelzoo/mobilenet)           | [CVPR 2018](https://arxiv.org/pdf/1801.04381.pdf)             | [Click](./ssseg/modules/models/backbones/mobilenet.py)   |
| MobileNetV3            | [Click](./docs/modelzoo/mobilenet)           | [ICCV 2019](https://arxiv.org/pdf/1905.02244.pdf)             | [Click](./ssseg/modules/models/backbones/mobilenet.py)   |
| SwinTransformer        | [Click](./docs/modelzoo/swin)                | [ICCV 2021](https://arxiv.org/pdf/2103.14030.pdf)             | [Click](./ssseg/modules/models/backbones/swin.py)        |
| VisionTransformer      | [Click](./docs/modelzoo/setr)                | [IClR 2021](https://arxiv.org/pdf/2010.11929.pdf)             | [Click](./ssseg/modules/models/backbones/vit.py)         |


#### Supported Segmentors

| Segmentor                         | Model Zoo                                    | Paper Link                                                                                                                                              | Code Snippet                                                                |
| :-:                               | :-:                                          | :-:                                                                                                                                                     | :-:                                                                         |
| FCN                               | [Click](./docs/modelzoo/fcn)                 | [TPAMI 2017](https://arxiv.org/pdf/1411.4038.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/fcn/fcn.py)                       |
| CE2P                              | [Click](./docs/modelzoo/ce2p)                | [AAAI 2019](https://arxiv.org/pdf/1809.05996.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/ce2p/ce2p.py)                     |
| SETR                              | [Click](./docs/modelzoo/setr)                | [CVPR 2021](https://arxiv.org/pdf/2012.15840.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/setr/setr.py)                     |
| ISNet                             | [Click](./docs/modelzoo/isnet)               | [ICCV 2021](https://arxiv.org/pdf/2108.12382.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/isnet/isnet.py)                   |
| ICNet                             | [Click](./docs/modelzoo/icnet)               | [ECCV 2018](https://arxiv.org/pdf/1704.08545.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/icnet/icnet.py)                   |
| CCNet                             | [Click](./docs/modelzoo/ccnet)               | [ICCV 2019](https://arxiv.org/pdf/1811.11721.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/ccnet/ccnet.py)                   |
| DANet                             | [Click](./docs/modelzoo/danet)               | [CVPR 2019](https://arxiv.org/pdf/1809.02983.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/danet/danet.py)                   |
| DMNet                             | [Click](./docs/modelzoo/dmnet)               | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf)        | [Click](./ssseg/modules/models/segmentors/dmnet/dmnet.py)                   |
| GCNet                             | [Click](./docs/modelzoo/gcnet)               | [TPAMI 2020](https://arxiv.org/pdf/1904.11492.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/gcnet/gcnet.py)                   |
| ISANet                            | [Click](./docs/modelzoo/isanet)              | [IJCV 2021](https://arxiv.org/pdf/1907.12273.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/isanet/isanet.py)                 |
| EncNet                            | [Click](./docs/modelzoo/encnet)              | [CVPR 2018](https://arxiv.org/pdf/1803.08904.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/encnet/encnet.py)                 |
| OCRNet                            | [Click](./docs/modelzoo/ocrnet)              | [ECCV 2020](https://arxiv.org/pdf/1909.11065.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/ocrnet/ocrnet.py)                 |
| DNLNet                            | [Click](./docs/modelzoo/dnlnet)              | [ECCV 2020](https://arxiv.org/pdf/2006.06668.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/dnlnet/dnlnet.py)                 |
| ANNNet                            | [Click](./docs/modelzoo/annnet)              | [ICCV 2019](https://arxiv.org/pdf/1908.07678.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/annnet/annnet.py)                 |
| EMANet                            | [Click](./docs/modelzoo/emanet)              | [ICCV 2019](https://arxiv.org/pdf/1907.13426.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/emanet/emanet.py)                 |
| PSPNet                            | [Click](./docs/modelzoo/pspnet)              | [CVPR 2017](https://arxiv.org/pdf/1612.01105.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/pspnet/pspnet.py)                 |
| PSANet                            | [Click](./docs/modelzoo/psanet)              | [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)                       | [Click](./ssseg/modules/models/segmentors/psanet/psanet.py)                 |
| APCNet                            | [Click](./docs/modelzoo/apcnet)              | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)   | [Click](./ssseg/modules/models/segmentors/apcnet/apcnet.py)                 |
| FastFCN                           | [Click](./docs/modelzoo/fastfcn)             | [ArXiv 2019](https://arxiv.org/pdf/1903.11816.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/fastfcn/fastfcn.py)               |
| UPerNet                           | [Click](./docs/modelzoo/upernet)             | [ECCV 2018](https://arxiv.org/pdf/1807.10221.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/upernet/upernet.py)               |
| PointRend                         | [Click](./docs/modelzoo/pointrend)           | [CVPR 2020](https://arxiv.org/pdf/1912.08193.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/pointrend/pointrend.py)           |
| Deeplabv3                         | [Click](./docs/modelzoo/deeplabv3)           | [ArXiv 2017](https://arxiv.org/pdf/1706.05587.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/deeplabv3/deeplabv3.py)           |
| Segformer                         | [Click](./docs/modelzoo/segformer)           | [NeurIPS 2021](https://arxiv.org/pdf/2105.15203.pdf)                                                                                                    | [Click](./ssseg/modules/models/segmentors/segformer/segformer.py)           |
| MaskFormer                        | [Click](./docs/modelzoo/maskformer)          | [NeurIPS 2021](https://arxiv.org/pdf/2107.06278.pdf)                                                                                                    | [Click](./ssseg/modules/models/segmentors/maskformer/maskformer.py)         |
| SemanticFPN                       | [Click](./docs/modelzoo/semanticfpn)         | [CVPR 2019](https://arxiv.org/pdf/1901.02446.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/semanticfpn/semanticfpn.py)       |
| NonLocalNet                       | [Click](./docs/modelzoo/nonlocalnet)         | [CVPR 2018](https://arxiv.org/pdf/1711.07971.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/nonlocalnet/nonlocalnet.py)       |
| Deeplabv3Plus                     | [Click](./docs/modelzoo/deeplabv3plus)       | [CVPR 2018](https://arxiv.org/pdf/1802.02611.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/deeplabv3plus/deeplabv3plus.py)   |
| MemoryNet-MCIBI                   | [Click](./docs/modelzoo/memorynet)           | [ICCV 2021](https://arxiv.org/pdf/2108.11819.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/memorynet/memorynet.py)           |
| MemoryNet-MCIBI++                 | [Click](./docs/modelzoo/memorynetv2)         | [TPAMI 2022](https://arxiv.org/pdf/2209.04471.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/memorynetv2/memorynetv2.py)       |
| Mixed Precision (FP16) Training   | [Click](./docs/modelzoo/fp16)                | [ArXiv 2017](https://arxiv.org/pdf/1710.03740.pdf)                                                                                                      | [Click](./ssseg/train.py)                                                   |


#### Supported Datasets

- [LIP](http://sysu-hcp.net/lip/)
- [ATR](http://sysu-hcp.net/lip/overview.php)
- [HRF](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/)
- [CIHP](http://sysu-hcp.net/lip/overview.php)
- [VSPW](https://www.vspwdataset.com/)
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
- [Dark Zurich](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip)
- [COCOStuff10k](https://github.com/nightrome/cocostuff10k)
- [COCOStuff164k](https://github.com/nightrome/cocostuff)
- [Pascal Context](https://cs.stanford.edu/~roozbeh/pascal-context/)
- [Nighttime Driving](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip)


## Citation

If you use this framework in your research, please cite this project:

```
@misc{ssseg2020,
    author = {Zhenchao Jin},
    title = {SSSegmentation: An Open Source Supervised Semantic Segmentation Toolbox Based on PyTorch},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/SegmentationBLWX/sssegmentation}},
}

@inproceedings{jin2021isnet,
    title={ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation},
    author={Jin, Zhenchao and Liu, Bin and Chu, Qi and Yu, Nenghai},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={7189--7198},
    year={2021}
}

@inproceedings{jin2021mining,
    title={Mining Contextual Information Beyond Image for Semantic Segmentation},
    author={Jin, Zhenchao and Gong, Tao and Yu, Dongdong and Chu, Qi and Wang, Jian and Wang, Changhu and Shao, Jie},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={7231--7241},
    year={2021}
}

@article{jin2022mcibi++,
    title={MCIBI++: Soft Mining Contextual Information Beyond Image for Semantic Segmentation},
    author={Jin, Zhenchao and Yu, Dongdong and Yuan, Zehuan and Yu, Lequan},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2022},
    publisher={IEEE}
}
```


## References

- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
