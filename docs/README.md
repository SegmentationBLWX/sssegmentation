# Introduction
```
sssegmentation is a general framework for our research on strongly supervised semantic segmentation.
```


# Supported
#### Supported Backbones
- [HRNet](https://arxiv.org/pdf/1908.07919.pdf)
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [ResNeSt](https://arxiv.org/pdf/2004.08955.pdf)
- [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)
- [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
#### Supported Models
- [FCN](https://arxiv.org/pdf/1411.4038.pdf)
- [CE2P](https://arxiv.org/pdf/1809.05996.pdf)
- [CCNet](https://arxiv.org/pdf/1811.11721.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983.pdf)
- [GCNet](https://arxiv.org/pdf/1904.11492.pdf)
- [DMNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf)
- [OCRNet](https://arxiv.org/pdf/1909.11065.pdf)
- [DNLNet](https://arxiv.org/pdf/2006.06668.pdf)
- [ANNNet](https://arxiv.org/pdf/1908.07678.pdf)
- [EMANet](https://arxiv.org/pdf/1907.13426.pdf)
- [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
- [PSANet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)
- [APCNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)
- [UPerNet](https://arxiv.org/pdf/1807.10221.pdf)
- [Deeplabv3](https://arxiv.org/pdf/1706.05587.pdf)
- [NonLocalNet](https://arxiv.org/pdf/1711.07971.pdf)
- [Deeplabv3Plus](https://arxiv.org/pdf/1802.02611.pdf)
#### Supported Datasets
- [LIP](http://sysu-hcp.net/lip/)
- [ATR](http://sysu-hcp.net/lip/overview.php)
- [CIHP](http://sysu-hcp.net/lip/overview.php)
- [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [MS COCO](https://cocodataset.org/#home)
- [CityScapes](https://www.cityscapes-dataset.com/)
- [Supervisely](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
- [SBUShadow](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)


# Quick Start
#### Build
You can build the project as follows:
```sh
cd ssseg/libs
sh make.sh
```
The datasets sssegmentation supports could be downloaded from here:
```
URL: https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw 
CODE: fn1i 
```
#### Train
You can train the models as follows:
```sh
usage:
sh scripts/train.sh ${CFGFILEPATH} [optional arguments]
or
sh scripts/distrain.sh ${NGPUS} ${CFGFILEPATH} [optional arguments]
```
Here is an example:
```sh
sh scripts/train.sh ssseg/cfgs/deeplabv3plus/cfgs_voc_resnet101os8.py
or
sh scripts/distrain.sh 4 ssseg/cfgs/deeplabv3plus/cfgs_voc_resnet101os8.py
```
#### Test
You can test the models as follows:
```sh
usage:
sh scripts/test.sh ${CFGFILEPATH} ${CHECKPOINTSPATH} [optional arguments]
or
sh scripts/distest.sh ${NGPUS} ${CFGFILEPATH} ${CHECKPOINTSPATH} [optional arguments]
```
Here is an example:
```sh
sh scripts/test.sh ssseg/cfgs/deeplabv3plus/cfgs_voc_resnet101os8.py deeplabv3plus_resnet101os8_voc_train/epoch_60.pth
or
sh scripts/distest.sh 4 ssseg/cfgs/deeplabv3plus/cfgs_voc_resnet101os8.py deeplabv3plus_resnet101os8_voc_train/epoch_60.pth
```
#### Demo
You can apply the models as follows:
```sh
usage: demo.py [-h] [--imagedir IMAGEDIR] [--imagepath IMAGEPATH] [--outputfilename OUTPUTFILENAME] --cfgfilepath CFGFILEPATH --checkpointspath CHECKPOINTSPATH

sssegmentation is a general framework for our research on strongly supervised semantic segmentation

optional arguments:
  -h, --help            show this help message and exit
  --imagedir IMAGEDIR   images dir for testing multi images
  --imagepath IMAGEPATH
                        imagepath for testing single image
  --outputfilename OUTPUTFILENAME
                        name to save output image(s)
  --cfgfilepath CFGFILEPATH
                        config file path you want to use
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to resume from
```
Here is an example:
```sh
python3 ssseg/demo.py --cfgfilepath ssseg/cfgs/deeplabv3plus/cfgs_voc_resnet101os8.py --checkpointspath deeplabv3plus_resnet101os8_voc_train/epoch_60.pth --imagepath testedimage.jpg
or
python3 ssseg/demo.py --cfgfilepath ssseg/cfgs/deeplabv3plus/cfgs_voc_resnet101os8.py --checkpointspath deeplabv3plus_resnet101os8_voc_train/epoch_60.pth --imagedir ./images
```