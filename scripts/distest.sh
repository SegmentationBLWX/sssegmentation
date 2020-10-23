#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..
NGPUS=$1
DATASETNAME=$2
BACKBONENAME=$3
MODELNAME=$4
CHECKPOINTSPATH=$5
NOEVAL=$6
python3 -m torch.distributed.launch --nproc_per_node $NGPUS ssseg/test.py --nproc_per_node $NGPUS --datasetname $DATASETNAME --backbonename $BACKBONENAME --modelname $MODELNAME --checkpointspath $CHECKPOINTSPATH ${6}