# Quick Start


## Train
#### Bash
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

#### Python
You can train the models as follows:
```sh
usage: train.py [-h] [--local_rank LOCAL_RANK]
                [--nproc_per_node NPROC_PER_NODE] --cfgfilepath CFGFILEPATH
                [--checkpointspath CHECKPOINTSPATH]

SSSegmentation is an open source strongly supervised semantic segmentation toolbox 
based on PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        node rank for distributed training
  --nproc_per_node NPROC_PER_NODE
                        number of process per node
  --cfgfilepath CFGFILEPATH
                        config file path you want to use
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to resume from
```


## Test
#### Bash
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

#### Python
You can test the models as follows:
```sh
usage: test.py [-h] [--local_rank LOCAL_RANK]
               [--nproc_per_node NPROC_PER_NODE] --cfgfilepath CFGFILEPATH
               [--evalmode EVALMODE] --checkpointspath CHECKPOINTSPATH

SSSegmentation is an open source strongly supervised semantic segmentation toolbox 
based on PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        node rank for distributed testing
  --nproc_per_node NPROC_PER_NODE
                        number of process per node
  --cfgfilepath CFGFILEPATH
                        config file path you want to use
  --evalmode EVALMODE   evaluate mode, support online and offline
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to resume from
```


## Inference
#### Python
You can apply the models as follows:
```sh
usage: inference.py [-h] [--imagedir IMAGEDIR] [--imagepath IMAGEPATH]
               [--outputfilename OUTPUTFILENAME] --cfgfilepath CFGFILEPATH
               --checkpointspath CHECKPOINTSPATH

SSSegmentation is an open source strongly supervised semantic segmentation toolbox 
based on PyTorch

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