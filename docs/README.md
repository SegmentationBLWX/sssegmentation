# Introduction
```
sssegmentation is a general framework for our research on strongly supervised semantic segmentation.
```


# Supported

## Supported Models
- [FCN](https://arxiv.org/pdf/1411.4038.pdf)
- [CE2P](https://arxiv.org/pdf/1809.05996.pdf)
- [OCRNet](https://arxiv.org/pdf/1909.11065.pdf)
- [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
- [Deeplabv3Plus](https://arxiv.org/pdf/1802.02611.pdf)

## Supported Backbones
- [HRNet](https://arxiv.org/pdf/1908.07919.pdf)
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [MobileNet](https://arxiv.org/pdf/1801.04381.pdf)

## Supported Datasets
- [LIP](http://sysu-hcp.net/lip/)
- [ATR](http://sysu-hcp.net/lip/overview.php)
- [CIHP](http://sysu-hcp.net/lip/overview.php)
- [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [MS COCO](https://cocodataset.org/#home)
- [CityScapes](https://www.cityscapes-dataset.com/)
- [Supervisely](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
- [SBUShadow](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)


# Usage

## Build
```sh
cd ssseg/libs
sh make.sh
```

## Train
#### Bash
```sh
usage:
sh scripts/train.sh ${DATASETNAME} ${BACKBONENAME} ${MODELNAME} [optional arguments]
or
sh scripts/distrain.sh ${NGPUS} ${DATASETNAME} ${BACKBONENAME} ${MODELNAME} [optional arguments]
example:
sh scripts/train.sh voc resnet101os16 deeplabv3plus
or
sh scripts/distrain.sh 4 voc resnet101os16 deeplabv3plus
```
#### Python
```sh
usage: train.py [-h] --modelname MODELNAME --datasetname DATASETNAME
                [--local_rank LOCAL_RANK] [--nproc_per_node NPROC_PER_NODE]
                --backbonename BACKBONENAME
                [--checkpointspath CHECKPOINTSPATH]

sssegmentation is a general framework for our research on strongly supervised semantic segmentation

optional arguments:
  -h, --help            show this help message and exit
  --modelname MODELNAME
                        model you want to train
  --datasetname DATASETNAME
                        dataset for training.
  --local_rank LOCAL_RANK
                        node rank for distributed training
  --nproc_per_node NPROC_PER_NODE
                        number of process per node
  --backbonename BACKBONENAME
                        backbone network for training.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to resume from.

example:
python3 ssseg/train.py --datasetname voc --backbonename resnet101os16 --modelname deeplabv3plus
or 
python3 -m torch.distributed.launch --nproc_per_node 4 ssseg/train.py --datasetname voc --backbonename resnet101os16 --modelname deeplabv3plus --nproc_per_node 4
```

## Test
#### Bash
```sh
usage:
sh scripts/test.sh ${DATASETNAME} ${BACKBONENAME} ${MODELNAME} ${CHECKPOINTSPATH} [optional arguments]
example:
sh scripts/test.sh voc resnet101os16 deeplabv3plus deeplabv3plus_resnet101os16_voc_train/epoch_60.pth
```
#### Python
```sh
usage: test.py [-h] --modelname MODELNAME --datasetname DATASETNAME
               [--local_rank LOCAL_RANK] [--nproc_per_node NPROC_PER_NODE]
               --backbonename BACKBONENAME [--noeval NOEVAL] --checkpointspath
               CHECKPOINTSPATH

sssegmentation is a general framework for our research on strongly supervised semantic segmentation

optional arguments:
  -h, --help            show this help message and exit
  --modelname MODELNAME
                        model you want to test
  --datasetname DATASETNAME
                        dataset for testing.
  --local_rank LOCAL_RANK
                        node rank for distributed testing
  --nproc_per_node NPROC_PER_NODE
                        number of process per node
  --backbonename BACKBONENAME
                        backbone network for testing.
  --noeval NOEVAL       set true if no ground truth could be used to eval the
                        results.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to resume from.

example: python3 ssseg/test.py --datasetname voc --backbonename resnet101os16 --modelname deeplabv3plus --checkpointspath deeplabv3plus_resnet101os16_voc_train/epoch_60.pth
```

## Demo
```sh
usage: demo.py [-h] [--imagedir IMAGEDIR] [--imagepath IMAGEPATH] --modelname
               MODELNAME --backbonename BACKBONENAME
               [--outputfilename OUTPUTFILENAME] --checkpointspath
               CHECKPOINTSPATH --datasetname DATASETNAME

sssegmentation is a general framework for our research on strongly supervised semantic segmentation

optional arguments:
  -h, --help            show this help message and exit
  --imagedir IMAGEDIR   images dir for testing multi images
  --imagepath IMAGEPATH
                        imagepath for testing single image
  --modelname MODELNAME
                        model you want to test
  --backbonename BACKBONENAME
                        backbone network for testing
  --outputfilename OUTPUTFILENAME
                        name to save output image(s)
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to resume from
  --datasetname DATASETNAME
                        dataset you used to train, for locating the config
                        filepath

example: python3 ssseg/demo.py --datasetname voc --backbonename resnet101os16 --modelname deeplabv3plus --checkpointspath deeplabv3plus_resnet101os16_voc_train/epoch_60.pth --imagepath testedimage.jpg
```