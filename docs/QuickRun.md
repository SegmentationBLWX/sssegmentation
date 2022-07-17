# Quick Run


## Train A Segmentor

SSSegmentation only supports distributed training which uses DistributedDataParallel.

All outputs (log files and checkpoints) will be saved to the working directory, which is specified by "work_dir" in the config file.

#### Train on a single machine

You can train the segmentors in a single machine as follows,

```sh
bash scripts/distrain.sh ${NGPUS} ${CFGFILEPATH} [optional arguments]
```

where "${NGPUS}" means the number of GPUS you want to use and "${CFGFILEPATH}" denotes for the config file path.
For example, you can train a segmentor on a single machine with the following commands,

```sh
bash scripts/distrain.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py
```

If you want to resume from the checkpoints, you can run as follows,

```sh
bash scripts/distrain.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py --checkpointspath annnet_resnet50os16_ade20k/epoch_44.pth
```

#### Train with multiple machines

Now, we only support training with multiple machines with Slurm.
Slurm is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use "slurmtrain.sh" to spawn training jobs.
It supports both single-node and multi-node training.

Specifically, you can train the segmentors with multiple machines as follows,

```sh
bash scripts/slurmtrain.sh ${PARTITION} ${JOBNAME} ${NGPUS} ${CFGFILEPATH} [optional arguments]
```

Here is an example of using 16 GPUs to train PSPNet on the dev partition,

```sh
bash scripts/slurmtrain.sh dev pspnet 16 ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py
```


## Test A Segmentor

We provide testing scripts to evaluate a whole dataset (Cityscapes, PASCAL VOC, ADE20k, etc.), and also some high-level apis for easier integration to other projects.

#### Test on a single machine

You can test the segmentors in a single machine as follows,

```sh
bash scripts/distest.sh ${NGPUS} ${CFGFILEPATH} ${CHECKPOINTSPATH} [optional arguments]
```

For example, you can test a segmentor on a single machine with the following commands,

```sh
bash scripts/distest.sh 4 ssseg/configs/annnet/annnet_resnet50os16_ade20k.py annnet_resnet50os16_ade20k/epoch_130.pth
```

#### Test with multiple machines

Now, we only support testing with multiple machines with Slurm.
Slurm is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use "slurmtest.sh" to spawn testing jobs.
It supports both single-node and multi-node testing.

Specifically, you can test the segmentors with multiple machines as follows,

```sh
bash scripts/slurmtest.sh ${PARTITION} ${JOBNAME} ${NGPUS} ${CFGFILEPATH} ${CHECKPOINTSPATH} [optional arguments]
```

Here is an example of using 16 GPUs to test PSPNet on the dev partition,

```sh
bash scripts/slurmtest.sh dev pspnet 16 ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/epoch_130.pth
```


## Inference A Segmentor

You can apply the segmentor as follows:

```sh
bash scripts/inference.sh ${CFGFILEPATH} ${CHECKPOINTSPATH} [optional arguments]
```

For example, if you want to inference one image, the command can be,

```sh
bash scripts/inference.sh ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/epoch_130.pth --imagepath dog.jpg
```

If you want to inference the images in one directory, the command can be,

```sh
bash scripts/inference.sh ssseg/configs/pspnet/pspnet_resnet101os8_ade20k.py pspnet_resnet101os8_ade20k/epoch_130.pth --imagedir dogs
```