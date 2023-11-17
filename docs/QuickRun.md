# Quick Run

In this chapter, we introduce some basic usage and commands of SSSegmentation to the users.


## Train A Segmentor

SSSegmentation only supports distributed training which uses `nn.parallel.DistributedDataParallel`.
All outputs (training log and checkpoints) will be saved to the working directory, which is specified by `SEGMENTOR_CFG['work_dir']` in the config file.

#### Train on a single machine

You can train the segmentors in a single machine as follows,

```sh
bash scripts/dist_train.sh ${NGPUS} ${CFGFILEPATH} [optional arguments]
```

where `${NGPUS}` means the number of GPUS you want to use and `${CFGFILEPATH}` denotes for the config file path.
For example, you can train a segmentor on a single machine with the following commands,

```sh
bash scripts/dist_train.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py
```

If you want to resume training from a specific checkpoint, you can run as follows,

```sh
bash scripts/dist_train.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py --ckptspath annnet_resnet50os16_ade20k/epoch_44.pth
```

If you want to auto resume from the last checkpoint, you can run as follows,

```sh
bash scripts/dist_train.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py --ckptspath annnet_resnet50os16_ade20k/epoch_last.pth
```

This trick is very useful when you are training your segmentors in an unstable environment and your program will be interrupted and restarted frequently.

#### Train with multiple machines

Now, we only support training with multiple machines using Slurm.
Slurm is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs.
It supports both single-node and multi-node training.

Specifically, you can train the segmentors with multiple machines as follows,

```sh
bash scripts/slurm_train.sh ${PARTITION} ${JOBNAME} ${NGPUS} ${CFGFILEPATH} [optional arguments]
```

Here is an example of using 16 GPUs to train PSPNet on the dev partition,

```sh
bash scripts/slurm_train.sh dev pspnet 16 ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py
```


## Test A Segmentor

We provide testing scripts to evaluate a whole dataset (Cityscapes, PASCAL VOC, ADE20k, etc.), and also some high-level apis for easier integration to other projects.
Also, all outputs (testing log and results) will be saved to the working directory, which is specified by `SEGMENTOR_CFG['work_dir']` in the config file.

#### Test on a single machine

You can test the segmentors in a single machine as follows,

```sh
bash scripts/dist_test.sh ${NGPUS} ${CFGFILEPATH} ${ckptspath} [optional arguments]
```

For example, you can test a segmentor on a single machine with the following commands,

```sh
bash scripts/dist_test.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py annnet_resnet50os16_ade20k/epoch_130.pth
```

#### Test with multiple machines

Now, we only support testing with multiple machines using Slurm.
Slurm is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_test.sh` to spawn testing jobs.
It supports both single-node and multi-node testing.

Specifically, you can test the segmentors with multiple machines as follows,

```sh
bash scripts/slurm_test.sh ${PARTITION} ${JOBNAME} ${NGPUS} ${CFGFILEPATH} ${ckptspath} [optional arguments]
```

Here is an example of using 16 GPUs to test PSPNet on the dev partition,

```sh
bash scripts/slurm_test.sh dev pspnet 16 ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/epoch_130.pth
```


## Inference A Segmentor

You can apply the trained segmentors to segment images with the following commands,

```sh
bash scripts/inference.sh ${CFGFILEPATH} ${ckptspath} [optional arguments]
```

For example, if you want to inference one image, you can run as follows,

```sh
bash scripts/inference.sh ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/epoch_130.pth --imagepath dog.jpg
```

If you want to inference the images in one directory, you can run as follows,

```sh
bash scripts/inference.sh ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/epoch_130.pth --imagedir dogs
```