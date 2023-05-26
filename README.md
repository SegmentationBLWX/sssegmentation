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

- [FCN](./docs/modelzoo/fcn)
- [CE2P](./docs/modelzoo/ce2p)
- [SETR](./docs/modelzoo/setr)
- [ISNet](./docs/modelzoo/isnet)
- [ICNet](./docs/modelzoo/icnet)
- [CCNet](./docs/modelzoo/ccnet)
- [DANet](./docs/modelzoo/danet)
- [GCNet](./docs/modelzoo/gcnet)
- [DMNet](./docs/modelzoo/dmnet)
- [ISANet](./docs/modelzoo/isanet)
- [EncNet](./docs/modelzoo/encnet)
- [OCRNet](./docs/modelzoo/ocrnet)
- [DNLNet](./docs/modelzoo/dnlnet)
- [ANNNet](./docs/modelzoo/annnet)
- [EMANet](./docs/modelzoo/emanet)
- [PSPNet](./docs/modelzoo/pspnet)
- [PSANet](./docs/modelzoo/psanet)
- [APCNet](./docs/modelzoo/apcnet)
- [FastFCN](./docs/modelzoo/fastfcn)
- [UPerNet](./docs/modelzoo/upernet)
- [PointRend](./docs/modelzoo/pointrend)
- [Deeplabv3](./docs/modelzoo/deeplabv3)
- [Segformer](./docs/modelzoo/segformer)
- [MaskFormer](./docs/modelzoo/maskformer)
- [SemanticFPN](./docs/modelzoo/semanticfpn)
- [NonLocalNet](./docs/modelzoo/nonlocalnet)
- [Deeplabv3Plus](./docs/modelzoo/deeplabv3plus)
- [MemoryNet-MCIBI](./docs/modelzoo/memorynet)
- [MemoryNet-MCIBI++](./docs/modelzoo/memorynetv2)
- [Mixed Precision (FP16) Training](./docs/modelzoo/fp16)

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
