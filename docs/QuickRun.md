# Quick Run

In this chapter, we introduce users to some basic usage and commands of SSSegmentation to help users quickly get started running SSSegmentation.


## Training and Testing Segmentors Integrated in SSSegmentation

SSSegmentation supports training and testing segmentation frameworks on a single machine or multiple machines (cluster) by utilizing `nn.parallel.DistributedDataParallel`.
In this section, you will learn how to train and test these supported segmentors using the scripts provided by SSSegmentation.

#### Training and Testing on A Single Machine

**1. Training a segmentor**

We provide `scripts/dist_train.sh` to launch training jobs on a single machine. The basic usage is as follows,

```sh
bash scripts/dist_train.sh ${NGPUS} ${CFGFILEPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${NGPUS}`: The number of GPUS you want to use,
- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `--ckptspath`: Checkpoints you want to resume from, if you want to resume from the latest checkpoint in the `SEGMENTOR_CFG['work_dir']` automatically, you can specify it as `f'{work_dir}/checkpoints-epoch-latest.pth'`,
- `--slurm`: Please add `--slurm` if you are using slurm to spawn training jobs.

Here, we provide some examples about training a segmentor on a single machine,

```sh
# train ANNNet
bash scripts/dist_train.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py
# load checkpoints-epoch-44.pth and then resume training ANNNet
bash scripts/dist_train.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py --ckptspath annnet_resnet50os16_ade20k/checkpoints-epoch-44.pth
# auto resume
bash scripts/dist_train.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py --ckptspath annnet_resnet50os16_ade20k/checkpoints-epoch-latest.pth
```

The last command will be very useful if you are training your segmentors in an unstable environment, *e.g.*, your program will be interrupted and restarted frequently.

**2. Testing a segmentor**

We provide `scripts/dist_test.sh` to launch testing jobs on a single machine. The basic usage is as follows,

```sh
bash scripts/dist_test.sh ${NGPUS} ${CFGFILEPATH} ${CKPTSPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${NGPUS}`: The number of GPUS you want to use,
- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `${CKPTSPATH}`: Checkpoints you want to resume from,
- `--eval_env`: Used to specify evaluate environment, support `server` environment (only save the test results which could be submitted to the corresponding dataset's official website to obtain the segmentation performance) and `local` environment (the default environment, test segmentors with the local images and annotations provided by the corresponding dataset),
- `--slurm`: Please add `--slurm` if you are using slurm to spawn testing jobs,
- `--ema`: Please add `--ema` if you want to load ema weights for segmentors.

Here, we provide some examples about testing a segmentor on a single machine,

```sh
# test ANNNet on ADE20k
bash scripts/dist_test.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py annnet_resnet50os16_ade20k/checkpoints-epoch-130.pth
# test ANNNet on Cityscapes
bash scripts/dist_test.sh 4 ssseg/configs/annnet/annnet_resnet50os16_cityscapes.py annnet_resnet50os16_cityscapes/checkpoints-epoch-220.pth
```

#### Training and Testing on Multiple Machines with Slurm

In SSSegmentation, we support training with multiple machines using Slurm, where Slurm is a good job scheduling system for computing clusters.

**1. Training a segmentor**

On a cluster managed by Slurm, you can use `scripts/slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training. The basic usage is as follows,

```sh
bash scripts/slurm_train.sh ${PARTITION} ${JOBNAME} ${NGPUS} ${CFGFILEPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${PARTITION}`: The Slurm partition you want to use,
- `${JOBNAME}`: The job name you want to show,
- `${NGPUS}`: The number of GPUS you want to use,
- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `--ckptspath`: Checkpoints you want to resume from, if you want to resume from the latest checkpoint in the `SEGMENTOR_CFG['work_dir']` automatically, you can specify it as `f'{work_dir}/checkpoints-epoch-latest.pth'`,
- `--slurm`: Please add `--slurm` if you are using slurm to spawn training jobs.

Here is an example of using 16 GPUs to train PSPNet on Slurm partition named *dev*,

```sh
bash scripts/slurm_train.sh dev pspnet 16 ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py --slurm
```

Please note that, `--slurm` is required to set for environment initialization if you are using slurm to spawn training jobs.

**2. Testing a segmentor**

Similar to the training task, SSSegmentation provides `scripts/slurm_test.sh` to spawn testing jobs. The basic usage is as follows,

```sh
bash scripts/slurm_test.sh ${PARTITION} ${JOBNAME} ${NGPUS} ${CFGFILEPATH} ${CKPTSPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${PARTITION}`: The Slurm partition you want to use,
- `${JOBNAME}`: The job name you want to show,
- `${NGPUS}`: The number of GPUS you want to use,
- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `${CKPTSPATH}`: Checkpoints you want to resume from,
- `--eval_env`: Used to specify evaluate environment, support `server` environment (only save the test results which could be submitted to the corresponding dataset's official website to obtain the segmentation performance) and `local` environment (the default environment, test segmentors with the local images and annotations provided by the corresponding dataset),
- `--slurm`: Please add `--slurm` if you are using slurm to spawn testing jobs,
- `--ema`: Please add `--ema` if you want to load ema weights for segmentors.

Here is an example of using 16 GPUs to test PSPNet on Slurm partition named *dev*,

```sh
bash scripts/slurm_test.sh dev pspnet 16 ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/checkpoints-epoch-130.pth --slurm
```

Similar to use slurm to spawn training jobs, `--slurm` is required to set for slurm environment initialization.

#### Training and Testing on Multiple Machines with AML

In SSSegmentation, we also support training with multiple machines using AML, where Azure Machine Learning (AML) is a cloud-based platform that enables data scientists and developers to build, train, and deploy machine learning models efficiently at scale. It offers end-to-end tools for automating workflows, managing experiments, and utilizing Azure’s compute resources for robust ML model training and deployment.

**1. Training a segmentor**

On a cluster managed by AML, you can use `scripts/aml_train.sh` to spawn training jobs in the pre-defined `job.yaml` file. It supports both single-node and multi-node training. The basic usage is as follows,

```sh
bash scripts/aml_train.sh ${CFGFILEPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `--ckptspath`: Checkpoints you want to resume from, if you want to resume from the latest checkpoint in the `SEGMENTOR_CFG['work_dir']` automatically, you can specify it as `f'{work_dir}/checkpoints-epoch-latest.pth'`,
- `--slurm`: Please add `--slurm` if you are using slurm to spawn training jobs.

Here is an example of spawning training jobs in the pre-defined `job.yaml` file,

```yaml
jobs:
  - name: fcn_resnet50os8_ade20k
    sku: 2xG8
    command:
      - bash scripts/aml_train.sh ssseg/configs/fcn/fcn_resnet50os8_ade20k.py
```

**2. Testing a segmentor**

Similar to the training task, SSSegmentation provides `scripts/aml_test.sh` to spawn testing jobs in the pre-defined `job.yaml` file. The basic usage is as follows,

```sh
bash scripts/aml_test.sh ${CFGFILEPATH} ${CKPTSPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `${CKPTSPATH}`: Checkpoints you want to resume from,
- `--eval_env`: Used to specify evaluate environment, support `server` environment (only save the test results which could be submitted to the corresponding dataset's official website to obtain the segmentation performance) and `local` environment (the default environment, test segmentors with the local images and annotations provided by the corresponding dataset),
- `--slurm`: Please add `--slurm` if you are using slurm to spawn testing jobs,
- `--ema`: Please add `--ema` if you want to load ema weights for segmentors.

Here is an example of spawning testing jobs in the pre-defined `job.yaml` file,

```sh
jobs:
  - name: fcn_resnet50os8_ade20k
    sku: 2xG8
    command:
      - bash scripts/aml_test.sh ssseg/configs/fcn/fcn_resnet50os8_ade20k.py fcn_resnet50os8_ade20k/checkpoints-epoch-130.pth
```


## Inference with Segmentors Integrated in SSSegmentation

SSSegmentation provides pre-trained models for semantic segmentation in [Model Zoo](https://sssegmentation.readthedocs.io/en/latest/ModelZoo.html), and supports multiple standard datasets, including Pascal VOC, Cityscapes, ADE20K, etc.
This section will show how to use existing pre-trained models to inference on given images. 

Specifically, SSSegmentation provides `scripts/inference.sh` to apply the trained segmentors to segment images. The basic usage is as follows,

```sh
bash scripts/inference.sh ${CFGFILEPATH} ${CKPTSPATH} [optional arguments]
```

This script accepts several optional arguments, including:

- `${CFGFILEPATH}`: The config file path which is used to customize segmentors,
- `${CKPTSPATH}`: Checkpoints you want to resume from,
- `--outputdir`: The directory used to save output image(s),
- `--imagepath`: Image path, which means we let the segmentor inference on the given image,
- `--imagedir`: Image directory, which means we let the segmentor inference on the images existed in the given image directory,
- `--ema`: Please add `--ema` if you want to load ema weights for segmentors.

Here are some example commands,

```sh
# inference on a given image
bash scripts/inference.sh ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/checkpoints-epoch-130.pth --imagepath dog.jpg
# inference on given images
bash scripts/inference.sh ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/checkpoints-epoch-130.pth --imagedir dogs
```

Please note that, if you specify `--imagedir` and `--imagepath` at the same time, only the value following `--imagedir` will be used.
And the image format should be in `[png, jpg, jpeg]`.