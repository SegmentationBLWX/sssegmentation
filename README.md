<div align="center">
  <img src="./docs/logo.png" width="600"/>
</div>
<br />

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://sssegmentation.readthedocs.io/en/latest/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sssegmentation)](https://pypi.org/project/sssegmentation/)
[![PyPI](https://img.shields.io/pypi/v/sssegmentation)](https://pypi.org/project/sssegmentation)
[![license](https://img.shields.io/github/license/SegmentationBLWX/sssegmentation.svg)](https://github.com/SegmentationBLWX/sssegmentation/blob/main/LICENSE)
[![PyPI - Downloads](https://pepy.tech/badge/sssegmentation)](https://pypi.org/project/sssegmentation/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sssegmentation?style=flat-square)](https://pypi.org/project/sssegmentation/)
[![issue resolution](https://isitmaintained.com/badge/resolution/SegmentationBLWX/sssegmentation.svg)](https://github.com/SegmentationBLWX/sssegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/SegmentationBLWX/sssegmentation.svg)](https://github.com/SegmentationBLWX/sssegmentation/issues)

Documents: https://sssegmentation.readthedocs.io/en/latest/


## What's New

- **2024-08-05**: Support [SAMV2](https://arxiv.org/pdf/2408.00714.pdf), refer to [inference-with-samv2](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-samv2) for more details.
- **2023-12-20**: Support [EdgeSAM](https://arxiv.org/pdf/2312.06660.pdf) and [SAMHQ](https://arxiv.org/pdf/2306.01567.pdf), refer to [inference-with-edgesam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-edgesam) and [inference-with-samhq](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-samhq) for more details.
- **2023-10-25**: Support [ConvNeXtV2](https://arxiv.org/pdf/2301.00808.pdf), refer to [Results and Models for ConvNeXtV2](./docs/modelzoo/convnextv2) for more details.
- **2023-10-23**: Support [MobileViT](https://arxiv.org/pdf/2110.02178.pdf) and [MobileViTV2](https://arxiv.org/pdf/2206.02680.pdf), refer to [Results and Models for MobileViT](./docs/modelzoo/mobilevit) for more details.
- **2023-10-18**: Support [Mask2Former](https://arxiv.org/pdf/2112.01527.pdf), refer to [Results and Models for Mask2Former](./docs/modelzoo/mask2former) for more details.
- **2023-10-17**: We release the source codes of [IDRNet: Intervention-Driven Relation Network for Semantic Segmentation](https://arxiv.org/pdf/2310.10755.pdf), which was accepted by NeurIPS 2023, refer to [Results and Models for IDRNet](./docs/modelzoo/idrnet) for more details.
- **2023-10-15**: Support [MobileSAM](https://arxiv.org/pdf/2306.14289.pdf), refer to [inference-with-mobilesam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-mobilesam) for more details.
- **2023-09-27**: Support [SAM](https://arxiv.org/pdf/2304.02643.pdf), refer to [inference-with-sam](https://sssegmentation.readthedocs.io/en/latest/AdvancedAPI.html#inference-with-sam) for more details.


## Introduction

SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.


## Major Features

- **High Performance**

  The performance of re-implemented segmentation algorithms is better than or comparable to other codebases.
 
- **Modular Design and Unified Benchmark**
  
  Various segmentation methods are unified into several specific modules.
  Benefiting from this design, SSSegmentation can integrate a great deal of popular and contemporary semantic segmentation frameworks and then, train and test them on unified benchmarks.
  
- **Fewer Dependencies**

  SSSegmenation tries its best to avoid introducing more dependencies when reproducing novel semantic segmentation approaches.
 

## Benchmark and Model Zoo

#### Supported Backbones

| Backbone               | Model Zoo                                    | Paper Link                                                    | Code Snippet                                             |
| :-:                    | :-:                                          | :-:                                                           | :-:                                                      |
| ConvNeXtV2             | [Click](./docs/modelzoo/convnextv2)          | [CVPR 2023](https://arxiv.org/pdf/2301.00808.pdf)             | [Click](./ssseg/modules/models/backbones/convnextv2.py)  |
| MobileViTV2            | [Click](./docs/modelzoo/mobilevit)           | [ArXiv 2022](https://arxiv.org/pdf/2206.02680.pdf)            | [Click](./ssseg/modules/models/backbones/mobilevit.py)   |
| ConvNeXt               | [Click](./docs/modelzoo/convnext)            | [CVPR 2022](https://arxiv.org/pdf/2201.03545.pdf)             | [Click](./ssseg/modules/models/backbones/convnext.py)    |
| MAE                    | [Click](./docs/modelzoo/mae)                 | [CVPR 2022](https://arxiv.org/pdf/2111.06377.pdf)             | [Click](./ssseg/modules/models/backbones/mae.py)         |
| MobileViT              | [Click](./docs/modelzoo/mobilevit)           | [ICLR 2022](https://arxiv.org/pdf/2110.02178.pdf)             | [Click](./ssseg/modules/models/backbones/mobilevit.py)   |
| BEiT                   | [Click](./docs/modelzoo/beit)                | [ICLR 2022](https://arxiv.org/pdf/2106.08254.pdf)             | [Click](./ssseg/modules/models/backbones/beit.py)        |
| Twins                  | [Click](./docs/modelzoo/twins)               | [NeurIPS 2021](https://arxiv.org/pdf/2104.13840.pdf)          | [Click](./ssseg/modules/models/backbones/twins.py)       |
| SwinTransformer        | [Click](./docs/modelzoo/swin)                | [ICCV 2021](https://arxiv.org/pdf/2103.14030.pdf)             | [Click](./ssseg/modules/models/backbones/swin.py)        |
| VisionTransformer      | [Click](./docs/modelzoo/setr)                | [IClR 2021](https://arxiv.org/pdf/2010.11929.pdf)             | [Click](./ssseg/modules/models/backbones/vit.py)         |
| BiSeNetV2              | [Click](./docs/modelzoo/bisenetv2)           | [IJCV 2021](https://arxiv.org/pdf/2004.02147.pdf)             | [Click](./ssseg/modules/models/backbones/bisenetv2.py)   |
| ResNeSt                | [Click](./docs/modelzoo/resnest)             | [ArXiv 2020](https://arxiv.org/pdf/2004.08955.pdf)            | [Click](./ssseg/modules/models/backbones/resnest.py)     |
| CGNet                  | [Click](./docs/modelzoo/cgnet)               | [TIP 2020](https://arxiv.org/pdf/1811.08201.pdf)              | [Click](./ssseg/modules/models/backbones/cgnet.py)       |
| HRNet                  | [Click](./docs/modelzoo/ocrnet)              | [CVPR 2019](https://arxiv.org/pdf/1908.07919.pdf)             | [Click](./ssseg/modules/models/backbones/hrnet.py)       |
| MobileNetV3            | [Click](./docs/modelzoo/mobilenet)           | [ICCV 2019](https://arxiv.org/pdf/1905.02244.pdf)             | [Click](./ssseg/modules/models/backbones/mobilenet.py)   |
| FastSCNN               | [Click](./docs/modelzoo/fastscnn)            | [ArXiv 2019](https://arxiv.org/pdf/1902.04502.pdf)            | [Click](./ssseg/modules/models/backbones/fastscnn.py)    |
| BiSeNetV1              | [Click](./docs/modelzoo/bisenetv1)           | [ECCV 2018](https://arxiv.org/pdf/1808.00897.pdf)             | [Click](./ssseg/modules/models/backbones/bisenetv1.py)   |
| MobileNetV2            | [Click](./docs/modelzoo/mobilenet)           | [CVPR 2018](https://arxiv.org/pdf/1801.04381.pdf)             | [Click](./ssseg/modules/models/backbones/mobilenet.py)   |
| ERFNet                 | [Click](./docs/modelzoo/erfnet)              | [T-ITS 2017](https://ieeexplore.ieee.org/document/8063438)    | [Click](./ssseg/modules/models/backbones/erfnet.py)      |
| ResNet                 | [Click](./docs/modelzoo/fcn)                 | [CVPR 2016](https://arxiv.org/pdf/1512.03385.pdf)             | [Click](./ssseg/modules/models/backbones/resnet.py)      |
| UNet                   | [Click](./docs/modelzoo/unet)                | [MICCAI 2015](https://arxiv.org/pdf/1505.04597.pdf)           | [Click](./ssseg/modules/models/backbones/unet.py)        |

#### Supported Segmentors

| Segmentor                         | Model Zoo                                    | Paper Link                                                                                                                                              | Code Snippet                                                                |
| :-:                               | :-:                                          | :-:                                                                                                                                                     | :-:                                                                         |
| SAMV2                             | [Click](./docs/modelzoo/samv2)               | [ArXiv 2024](https://arxiv.org/pdf/2408.00714.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/samv2/samv2.py)                   |
| EdgeSAM                           | [Click](./docs/modelzoo/edgesam)             | [ArXiv 2023](https://arxiv.org/pdf/2312.06660.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/edgesam/edgesam.py)               |
| IDRNet                            | [Click](./docs/modelzoo/idrnet)              | [NeurIPS 2023](https://arxiv.org/pdf/2310.10755.pdf)                                                                                                    | [Click](./ssseg/modules/models/segmentors/idrnet/idrnet.py)                 |
| MobileSAM                         | [Click](./docs/modelzoo/mobilesam)           | [ArXiv 2023](https://arxiv.org/pdf/2306.14289.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/mobilesam/mobilesam.py)           |
| SAMHQ                             | [Click](./docs/modelzoo/samhq)               | [NeurIPS 2023](https://arxiv.org/pdf/2306.01567.pdf)                                                                                                    | [Click](./ssseg/modules/models/segmentors/samhq/samhq.py)                   |
| SAM                               | [Click](./docs/modelzoo/sam)                 | [ArXiv 2023](https://arxiv.org/pdf/2304.02643.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/sam/sam.py)                       |
| MCIBI++                           | [Click](./docs/modelzoo/mcibiplusplus)       | [TPAMI 2022](https://arxiv.org/pdf/2209.04471.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/mcibiplusplus/mcibiplusplus.py)   |
| Mask2Former                       | [Click](./docs/modelzoo/mask2former)         | [CVPR 2022](https://arxiv.org/pdf/2112.01527.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/mask2former/mask2former.py)       |
| ISNet                             | [Click](./docs/modelzoo/isnet)               | [ICCV 2021](https://arxiv.org/pdf/2108.12382.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/isnet/isnet.py)                   |
| MCIBI                             | [Click](./docs/modelzoo/mcibi)               | [ICCV 2021](https://arxiv.org/pdf/2108.11819.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/mcibi/mcibi.py)                   |
| MaskFormer                        | [Click](./docs/modelzoo/maskformer)          | [NeurIPS 2021](https://arxiv.org/pdf/2107.06278.pdf)                                                                                                    | [Click](./ssseg/modules/models/segmentors/maskformer/maskformer.py)         |
| Segformer                         | [Click](./docs/modelzoo/segformer)           | [NeurIPS 2021](https://arxiv.org/pdf/2105.15203.pdf)                                                                                                    | [Click](./ssseg/modules/models/segmentors/segformer/segformer.py)           |
| SETR                              | [Click](./docs/modelzoo/setr)                | [CVPR 2021](https://arxiv.org/pdf/2012.15840.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/setr/setr.py)                     |
| ISANet                            | [Click](./docs/modelzoo/isanet)              | [IJCV 2021](https://arxiv.org/pdf/1907.12273.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/isanet/isanet.py)                 |
| DNLNet                            | [Click](./docs/modelzoo/dnlnet)              | [ECCV 2020](https://arxiv.org/pdf/2006.06668.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/dnlnet/dnlnet.py)                 |
| PointRend                         | [Click](./docs/modelzoo/pointrend)           | [CVPR 2020](https://arxiv.org/pdf/1912.08193.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/pointrend/pointrend.py)           |
| OCRNet                            | [Click](./docs/modelzoo/ocrnet)              | [ECCV 2020](https://arxiv.org/pdf/1909.11065.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/ocrnet/ocrnet.py)                 |
| GCNet                             | [Click](./docs/modelzoo/gcnet)               | [TPAMI 2020](https://arxiv.org/pdf/1904.11492.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/gcnet/gcnet.py)                   |
| APCNet                            | [Click](./docs/modelzoo/apcnet)              | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)   | [Click](./ssseg/modules/models/segmentors/apcnet/apcnet.py)                 |
| DMNet                             | [Click](./docs/modelzoo/dmnet)               | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf)        | [Click](./ssseg/modules/models/segmentors/dmnet/dmnet.py)                   |
| ANNNet                            | [Click](./docs/modelzoo/annnet)              | [ICCV 2019](https://arxiv.org/pdf/1908.07678.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/annnet/annnet.py)                 |
| EMANet                            | [Click](./docs/modelzoo/emanet)              | [ICCV 2019](https://arxiv.org/pdf/1907.13426.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/emanet/emanet.py)                 |
| FastFCN                           | [Click](./docs/modelzoo/fastfcn)             | [ArXiv 2019](https://arxiv.org/pdf/1903.11816.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/fastfcn/fastfcn.py)               |
| SemanticFPN                       | [Click](./docs/modelzoo/semanticfpn)         | [CVPR 2019](https://arxiv.org/pdf/1901.02446.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/semanticfpn/semanticfpn.py)       |
| CCNet                             | [Click](./docs/modelzoo/ccnet)               | [ICCV 2019](https://arxiv.org/pdf/1811.11721.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/ccnet/ccnet.py)                   |
| CE2P                              | [Click](./docs/modelzoo/ce2p)                | [AAAI 2019](https://arxiv.org/pdf/1809.05996.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/ce2p/ce2p.py)                     |
| DANet                             | [Click](./docs/modelzoo/danet)               | [CVPR 2019](https://arxiv.org/pdf/1809.02983.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/danet/danet.py)                   |
| PSANet                            | [Click](./docs/modelzoo/psanet)              | [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)                       | [Click](./ssseg/modules/models/segmentors/psanet/psanet.py)                 |
| UPerNet                           | [Click](./docs/modelzoo/upernet)             | [ECCV 2018](https://arxiv.org/pdf/1807.10221.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/upernet/upernet.py)               |
| EncNet                            | [Click](./docs/modelzoo/encnet)              | [CVPR 2018](https://arxiv.org/pdf/1803.08904.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/encnet/encnet.py)                 |
| Deeplabv3Plus                     | [Click](./docs/modelzoo/deeplabv3plus)       | [ECCV 2018](https://arxiv.org/pdf/1802.02611.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/deeplabv3plus/deeplabv3plus.py)   |
| NonLocalNet                       | [Click](./docs/modelzoo/nonlocalnet)         | [CVPR 2018](https://arxiv.org/pdf/1711.07971.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/nonlocalnet/nonlocalnet.py)       |
| ICNet                             | [Click](./docs/modelzoo/icnet)               | [ECCV 2018](https://arxiv.org/pdf/1704.08545.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/icnet/icnet.py)                   |
| Mixed Precision (FP16) Training   | [Click](./docs/modelzoo/fp16)                | [ArXiv 2017](https://arxiv.org/pdf/1710.03740.pdf)                                                                                                      | [Click](./ssseg/train.py)                                                   |
| Deeplabv3                         | [Click](./docs/modelzoo/deeplabv3)           | [ArXiv 2017](https://arxiv.org/pdf/1706.05587.pdf)                                                                                                      | [Click](./ssseg/modules/models/segmentors/deeplabv3/deeplabv3.py)           |
| PSPNet                            | [Click](./docs/modelzoo/pspnet)              | [CVPR 2017](https://arxiv.org/pdf/1612.01105.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/pspnet/pspnet.py)                 |
| FCN                               | [Click](./docs/modelzoo/fcn)                 | [TPAMI 2017](https://arxiv.org/pdf/1411.4038.pdf)                                                                                                       | [Click](./ssseg/modules/models/segmentors/fcn/fcn.py)                       |

#### Supported Datasets

| Dataset                | Project Link                                                                               | Paper Link                                                                                                         | Code Snippet                                             | Download Script                                                                                                                 |
| :-:                    | :-:                                                                                        | :-:                                                                                                                | :-:                                                      | :-:                                                                                                                             |
| VSPW                   | [Click](https://www.vspwdataset.com/)                                                      | [CVPR 2021](https://yu-wu.net/pdf/CVPR21-vspw.pdf)                                                                 | [Click](./ssseg/modules/datasets/vspw.py)                | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh vspw` </details>                                              |
| Supervisely            | [Click](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)   | [Website Release 2020](https://ecosystem.supervisely.com/projects/persons)                                         | [Click](./ssseg/modules/datasets/supervisely.py)         | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh supervisely` </details>                                       |
| Dark Zurich            | [Click](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip) | [ICCV 2019](https://arxiv.org/pdf/1901.05946.pdf)                                                                  | [Click](./ssseg/modules/datasets/darkzurich.py)          | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh darkzurich` </details>                                        |
| Nighttime Driving      | [Click](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip)      | [ITSC 2018](https://arxiv.org/pdf/1810.02575.pdf)                                                                  | [Click](./ssseg/modules/datasets/nighttimedriving.py)    | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh nighttimedriving` </details>                                  |
| CIHP                   | [Click](http://sysu-hcp.net/lip/overview.php)                                              | [ECCV 2018](https://arxiv.org/pdf/1808.00157.pdf)                                                                  | [Click](./ssseg/modules/datasets/cihp.py)                | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh cihp` </details>                                              |
| COCOStuff10k           | [Click](https://github.com/nightrome/cocostuff10k)                                         | [CVPR 2018](https://arxiv.org/pdf/1612.03716.pdf)                                                                  | [Click](./ssseg/modules/datasets/coco.py)                | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh cocostuff10k` </details>                                      |
| COCOStuff164k          | [Click](https://github.com/nightrome/cocostuff)                                            | [CVPR 2018](https://arxiv.org/pdf/1612.03716.pdf)                                                                  | [Click](./ssseg/modules/datasets/coco.py)                | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh coco` </details>                                              |
| MHPv1&v2               | [Click](https://lv-mhp.github.io/dataset)                                                  | [ArXiv 2017](https://arxiv.org/pdf/1705.07206.pdf)                                                                 | [Click](./ssseg/modules/datasets/mhp.py)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh mhpv1` & `bash scripts/prepare_datasets.sh mhpv2` </details>  |
| LIP                    | [Click](http://sysu-hcp.net/lip/)                                                          | [CVPR 2017](https://arxiv.org/pdf/1703.05446.pdf)                                                                  | [Click](./ssseg/modules/datasets/lip.py)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh lip` </details>                                               |
| ADE20k                 | [Click](https://groups.csail.mit.edu/vision/datasets/ADE20K/)                              | [CVPR 2017](https://arxiv.org/pdf/1608.05442.pdf)                                                                  | [Click](./ssseg/modules/datasets/ade20k.py)              | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh ade20k` </details>                                            |
| SBUShadow              | [Click](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)        | [ECCV 2016](https://www3.cs.stonybrook.edu/~cvl/content/papers/2016/LSS_ECCV16.pdf?)                               | [Click](./ssseg/modules/datasets/sbushadow.py)           | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh sbushadow` </details>                                         |
| CityScapes             | [Click](https://www.cityscapes-dataset.com/)                                               | [CVPR 2016](https://arxiv.org/pdf/1604.01685.pdf)                                                                  | [Click](./ssseg/modules/datasets/cityscapes.py)          | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh cityscapes` </details>                                        |
| ATR                    | [Click](http://sysu-hcp.net/lip/overview.php)                                              | [ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/papers/Liang_Human_Parsing_With_ICCV_2015_paper.pdf)   | [Click](./ssseg/modules/datasets/atr.py)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh atr` </details>                                               |
| Pascal Context         | [Click](https://cs.stanford.edu/~roozbeh/pascal-context/)                                  | [CVPR 2014](https://cs.stanford.edu/~roozbeh/pascal-context/mottaghi_et_al_cvpr14.pdf)                             | [Click](./ssseg/modules/datasets/voc.py)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh pascalcontext` </details>                                     |
| MS COCO                | [Click](https://cocodataset.org/#home)                                                     | [ECCV 2014](https://arxiv.org/pdf/1405.0312.pdf)                                                                   | [Click](./ssseg/modules/datasets/coco.py)                | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh coco` </details>                                              |
| HRF                    | [Click](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/)                 | [Int J Biomed Sci 2013](https://www.hindawi.com/journals/ijbi/2013/154860/)                                        | [Click](./ssseg/modules/datasets/hrf.py)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh hrf` </details>                                               |
| CHASE DB1              | [Click](https://staffnet.kingston.ac.uk/~ku15565/)                                         | [TBME 2012](https://ieeexplore.ieee.org/document/6224174)                                                          | [Click](./ssseg/modules/datasets/chasedb1.py)            | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh chase_db1` </details>                                         |
| PASCAL VOC             | [Click](http://host.robots.ox.ac.uk/pascal/VOC/)                                           | [IJCV 2010](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)                                          | [Click](./ssseg/modules/datasets/voc.py)                 | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh pascalvoc` </details>                                         |
| DRIVE                  | [Click](https://drive.grand-challenge.org/)                                                | [TMI 2004](https://ieeexplore.ieee.org/document/1282003)                                                           | [Click](./ssseg/modules/datasets/drive.py)               | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh drive` </details>                                             |
| STARE                  | [Click](http://cecas.clemson.edu/~ahoover/stare/)                                          | [TMI 2000](https://ieeexplore.ieee.org/document/845178)                                                            | [Click](./ssseg/modules/datasets/stare.py)               | <details><summary>CMD</summary> `bash scripts/prepare_datasets.sh stare` </details>                                             |


## Citation

If you use SSSegmentation in your research, please consider citing this project,

```
@article{jin2023sssegmenation,
    title={SSSegmenation: An Open Source Supervised Semantic Segmentation Toolbox Based on PyTorch},
    author={Jin, Zhenchao},
    journal={arXiv preprint arXiv:2305.17091},
    year={2023}
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

@inproceedings{jin2023idrnet,
    title={IDRNet: Intervention-Driven Relation Network for Semantic Segmentation},
    author={Jin, Zhenchao and Hu, Xiaowei and Zhu, Lingting and Song, Luchuan and Yuan, Li and Yu, Lequan},
    booktitle={Thirty-Seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```


## References

We are very grateful to the following projects for their help in building SSSegmentation,

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former/)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [apex](https://github.com/NVIDIA/apex)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [VSPW_code](https://github.com/VSPW-dataset/VSPW_code)
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain)
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)
- [sam-hq](https://github.com/SysCV/sam-hq)
- [segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
